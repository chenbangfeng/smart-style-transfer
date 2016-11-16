require 'torch'
require 'nn'
require 'image'
require 'optim'
require 'paths'
require 'loadcaffe'


local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style_image', 'samples/TheWave-Style.jpg',
           'Style target image')
cmd:option('-content_image', 'tests/origami.jpeg',
           'Content target image')
cmd:option('-image_size', 512, 'Maximum height / width of generated image')
cmd:option('-cpu', true, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-mask_labels', 'masks/origami.dat',
           'Labels to generate masks for smarter style transfer')

-- Optimization options
cmd:option('-content_weight', 5e0)
cmd:option('-style_weight', 1e2)
cmd:option('-tv_weight', 1e-3)
cmd:option('-num_iterations', 1000)
cmd:option('-normalize_gradients', false)
cmd:option('-init', 'random', 'random|image')
cmd:option('-optimizer', 'lbfgs', 'lbfgs|adam|sgd')
cmd:option('-learning_rate', 1e1)

-- Output options
cmd:option('-print_iter', 20)
cmd:option('-save_iter', 10)
cmd:option('-output_image', 'out.png')
cmd:option('-output_dir',      'frames', 'Output directory to save to.' )

-- Other options
cmd:option('-style_scale', 1.0)
cmd:option('-original_colors', 0)
cmd:option('-pooling', 'avg', 'max|avg')
cmd:option('-proto_file', 'models/VGG_ILSVRC_19_layers_deploy.prototxt')
cmd:option('-model_file', 'models/VGG_ILSVRC_19_layers.caffemodel')
cmd:option('-backend', 'nn', 'nn|cudnn|clnn')
cmd:option('-cudnn_autotune', false)
cmd:option('-seed', -1)

cmd:option('-content_layers', 'relu4_2', 'layers for content')
cmd:option('-style_layers', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1', 'layers for style')

paths.dofile('util/nnModule.lua')
paths.dofile('layers/contentLoss.lua')
paths.dofile('layers/styleLoss.lua')
paths.dofile('layers/totalLoss.lua')
paths.dofile('layers/gramNetwork.lua')
paths.dofile('images.lua')

local function main(params)

  local frames_dir = params.output_dir
  if not paths.dirp(frames_dir) then
      paths.mkdir(frames_dir)
  end

  if not params.cpu then
    require 'cutorch'
    if params.backend ~= 'clnn' then
      require 'cunn'
    else
      require 'clnn'
    end
  else
    params.backend = 'nn'
  end

  if params.backend == 'cudnn' then
    require 'cudnn'
    if params.cudnn_autotune then
      cudnn.benchmark = true
    end
    cudnn.SpatialConvolution.accGradParameters = nn.SpatialConvolutionMM.accGradParameters -- ie: nop
  end
  
  local loadcaffe_backend = params.backend
  if params.backend == 'clnn' then loadcaffe_backend = 'nn' end
  local cnn = loadcaffe.load(params.proto_file, params.model_file, loadcaffe_backend):float()
  if not params.cpu then
    if params.backend ~= 'clnn' then
      cnn:cuda()
    else
      cnn:cl()
    end
  end
  
  -- Load content image 
  local content_image = image.load(params.content_image, 3)
  content_image = image.scale(content_image, params.image_size, 'bilinear')
  local content_image_caffe = preprocess(content_image):float()

  -- Load style image
  local style_size = math.ceil(params.style_scale * params.image_size)
  local style_img = image.load(params.style_image, 3)
  style_img = image.scale(style_img, style_size, 'bilinear')
  local style_image_caffe = preprocess(style_img):float()

  -- Load mask labels
  local mask_labels = torch.load(params.mask_labels)
  local max = mask_labels:max()
  local min = mask_labels:min()
  -- Normalize so we can scale this like an image
  mask_labels:csub(min):div(max-min)
  mask_labels = image.scale(mask_labels, params.image_size, 'bilinear')
  -- Undo normalization
  mask_labels:mul(max-min):add(min):round()

  -- Normalize the style blending weights so they sum to 1
  style_weights = {
    ['conv1_1'] = 1,
    ['conv2_1'] = 1,
    ['conv3_1'] = 1,
    ['conv4_1'] = 1,
    ['conv5_1'] = 1,
  }
  local style_sum = 0
  for i = 1, #style_weights do
    style_weights[i] = tonumber(style_weights[i])
    style_sum = style_sum + style_weights[i]
  end

  for i = 1, #style_weights do
    style_weights[i] = style_weights[i] / style_sum
  end
  

  if not params.cpu then
    if params.backend ~= 'clnn' then
      content_image_caffe = content_image_caffe:cuda()
      style_image_caffe = style_image_caffe:cuda()
    else
      content_image_caffe = content_image_caffe:cl()
      style_image_caffe = style_image_caffe:cl()
    end
  end
  
  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")

  -- Set up the network, inserting style and content loss modules
  local content_losses, style_losses = {}, {}
  local next_content_idx, next_style_idx = 1, 1
  local net = nn.Sequential()


  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    if not params.cpu then
      if params.backend ~= 'clnn' then
        tv_mod:cuda()
      else
        tv_mod:cl()
      end
    end
    net:add(tv_mod)
  end

  local loss_layers = {}
  local masks_weight = {}
  local masks_sums = {}
  local masks_min= {}
  local masks_max= {}
  for i = 1, #cnn do
    if next_content_idx <= #content_layers or next_style_idx <= #style_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')


      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        if not params.cpu then
          if params.backend ~= 'clnn' then
            avg_pool_layer:cuda()
          else
            avg_pool_layer:cl()
          end
        end
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end


      if name == content_layers[next_content_idx] then
        print("Setting up content layer", i, ":", layer.name)
        local target = net:forward(content_image_caffe):clone()
        local norm = params.normalize_gradients
        local loss_module = nn.ContentLoss(params.content_weight, target, norm):float()
        loss_module:name(name)
        if not params.cpu then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(content_losses, loss_module)
        next_content_idx = next_content_idx + 1
      end


      if name == style_layers[next_style_idx] then
        print("Setting up style layer  ", i, ":", layer.name)
        local style_gram = GramMatrix():float()
        if not params.cpu then
          if params.backend ~= 'clnn' then
            style_gram = style_gram:cuda()
          else
            style_gram = style_gram:cl()
          end
        end
        local target = nil
        local tests = nill
        local target_features = net:forward(style_image_caffe):clone()
        local target_i = style_gram:forward(target_features):clone()

        --Init the total of the masks for the weight
        local total_mask=torch.Tensor(mask_labels:size()):fill(0)
        -- The segements in the label are label starting with zero
        for i=0, mask_labels:max(), 1 do

          if not masks_sums[i+1] then
            masks_sums[i+1] = 0
          end

          if not masks_min[i+1] then
            masks_min[i+1] = 0
          end

          if not masks_max[i+1] then
            masks_max[i+1] = 0
          end
          -- Need a network to get the gram matrix
          local mask_gram = GramMatrix():float()
          if not params.cpu then
            if params.backend ~= 'clnn' then
              mask_gram = mask_gram:cuda()
            else
              mask_gram = mask_gram:cl()
            end
          end

          -- Copy the mask_label so we don't modify it
          local mask = mask_labels:clone()
          local img2 = content_image_caffe:clone()

          -- Make the actual mask
          mask = mask:apply(function(val)
            if val == i then 
              return 1
            else
              return 0
            end
          end)

          -- Apply the mask to the img
          local numChannels = img2:size(1)
          for i=1, numChannels, 1 do
            img2[i]:cmul(mask)
          end
          local mask_features = net:forward(img2):clone()
          -- Feed the masked image into the gram matrix
          local mask_target = mask_gram:forward(mask_features):clone()
          mask_target:div(img2:nElement())

          local crit = nn.MSECriterion()

          -- Use the mean squared error as the distance metric
          local MSE = crit:forward(mask_target, target_i)
          masks_sums[i+1] = masks_sums[i+1] + MSE

          if MSE < masks_min[i+1] then
            masks_min[i+1] = MSE
          end

          if MSE > masks_max[i+1] then
            masks_max[i+1] = MSE
          end

           -- Make the actual mask
          mask = mask:apply(function(val)
            if val == 1 then 
              return MSE
            else
              return 0
            end
          end)

          -- Add the mask values to the total mask
          total_mask:add(mask:double())
          collectgarbage()
        end

        -- Make the mask layer for the weights
        masks_weight[name] = total_mask
        table.insert(loss_layers, i)
        target_i:div(target_features:nElement())
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target_i, norm):float()
        loss_module:name(name)
        if not params.cpu then
          if params.backend ~= 'clnn' then
            loss_module:cuda()
          else
            loss_module:cl()
          end
        end
        net:add(loss_module)
        table.insert(style_losses, loss_module)
        next_style_idx = next_style_idx + 1
      end
    end
    collectgarbage()
  end
    -- We don't need the base CNN anymore, so clean it up to save memory.
  cnn = nil
  for i=1,#net.modules do
    local module = net.modules[i]
    if torch.type(module) == 'nn.SpatialConvolutionMM' then
        -- remove these, not used, but uses gpu memory
        module.gradWeight = nil
        module.gradBias = nil
    end
  end
  print(collectgarbage('count')*1024)
  collectgarbage()
  print('here')
  -- We have the mask, now we need to normalizes across the layers for a segment
  min_masks = {}
  diff_masks = {}
  for i,name in pairs(style_layers) do

    min_masks[name] = torch.Tensor(mask_labels:size()):fill(0)
    diff_masks[name] = torch.Tensor(mask_labels:size()):fill(0)
    for i=0, mask_labels:max(), 1 do

      local total = masks_sums[i+1]
      local min = masks_min[i+1]
      local max = masks_max[i+1]

      -- Copy the mask_label so we don't modify it
      local mask = mask_labels:clone()
      local img2 = content_image_caffe:clone()

      -- Make the actual mask
      min_mask = mask:apply(function(val)
        if val == i then 
          return min
        else
          return 0
        end
      end):double()

      min_masks[name]:add(min_mask)
      -- Make the actual mask
      max_mask = mask:apply(function(val)
        if val == i then 
          return max
        else
          return 0
        end
      end):double()

      diff_masks[name]:add(max_mask):csub(min_mask)
    end

    -- Normalize segments across layers instead of within the layers
    masks_weight[name] = masks_weight[name]:double():csub(min_masks[name]):cdiv(diff_masks[name])
  end

  for i, loss_module in ipairs(style_losses) do 
    print(i, loss_module._name)
  end
  
  -- Initialize the image
  if params.seed >= 0 then
    torch.manualSeed(params.seed)
  end
  local img = nil
  if params.init == 'random' then
    img = torch.randn(content_image:size()):float():mul(0.001)
  elseif params.init == 'image' then
    img = content_image_caffe:clone():float()
  else
    error('Invalid init type')
  end
  if not params.cpu then
    if params.backend ~= 'clnn' then
      img = img:cuda()
    else
      img = img:cl()
    end
  end
  
  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = params.num_iterations,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
  elseif params.optimizer == 'sgd' then
    optim_state = {
        maxIter = params.num_iterations,
        momentum = 0.9,
        dampening = 0.0,
        learningRate = 1e-3
    }
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss)
    local verbose = (params.print_iter > 0 and t % params.print_iter == 0)
    if verbose then
      print(string.format('Iteration %d / %d', t, params.num_iterations))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
      end
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function maybe_save(t)
    local should_save = params.save_iter > 0 and t <= 20 or t % params.save_iter == 0
    should_save = should_save or t == params.num_iterations
    if should_save then
      local disp = deprocess(img:double())
      disp = image.minmax{tensor=disp, min=0, max=1}
      local filename = build_filename(paths.concat(frames_dir, params.output_image), t)
      if t == params.num_iterations then
        filename = params.output_image
      end

      -- Maybe perform postprocessing for color-independent style transfer
      if params.original_colors == 1 then
        disp = original_colors(content_image, disp)
      end

      image.save(filename, disp)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this function many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:updateGradInput(x, dy)
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end
    maybe_print(num_calls, loss)
    maybe_save(num_calls)

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = optim.lbfgs(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, params.num_iterations do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end
end
  

function build_filename(output_image, iteration)
  local ext = paths.extname(output_image)
  local basename = paths.basename(output_image, ext)
  local directory = paths.dirname(output_image)
  return string.format('%s/%s_%d.%s',directory, basename, iteration, ext)
end

local params = cmd:parse(arg)
main(params)