-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0
  self.gram = GramMatrix()
  self.G = nil
  self.mask = nil
  self.normalizer = 1
  self.crit = nn.MSECriterion()
end

function StyleLoss:setMask(mask)
  self.mask = mask
end

function StyleLoss:setNormalizer(normalizer)
  self.normalizer = normalizer
end

function StyleLoss:maskInput(input)
  if not input then print('input is nil') end
  local clone = input:clone():view(input:size(1), -1)
  --clone[1]:cmul(self.mask)
  for i=1, clone:size(1), 1 do
    clone[i]:cmul(self.mask)
  end
  return clone:view(input:size())
end

function StyleLoss:updateOutput(input)
  if self.mask and false then
    self.G = self.gram:forward(self:maskInput(input))
    self.loss = self.crit:forward(self.G, self.target)
  else
    self.G = self.gram:forward(input):div(input:nElement())
    self.loss = self.crit:forward(self.G, self.target)
  end
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG
  if self.mask and false then
    dG = self.crit:backward(self.G, self.target)
    self.gradInput = self.gram:backward(input, dG:div(self.normalizer))
  else
    dG = self.crit:backward(self.G, self.target):div(input:nElement())
    self.gradInput = self.gram:backward(input, dG)
  end

  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end
