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
  local clone = input:clone()
  for i=1, clone:size(1), 1 do
    clone:view(clone:size(1), clone:size(2) * clone:size(3))[i]:cmul(self.mask:float())
  end
  return clone
end

function StyleLoss:updateOutput(input)
  if self.mask then
    self.G = self.gram:forward(self:maskInput(input))
  else
    self.G = self.gram:forward(input)
  end
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target:div(self.normalizer))
  self.loss = self.loss * self.strength
  self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  if self.mask then
    self.G = self.gram:forward(self:maskInput(input))
  else
    self.G = self.gram:forward(input)
  end
  local dG = self.crit:backward(self.G, self.target:div(self.normalizer))
  dG:div(input:nElement())

  self.gradInput = self.gram:backward(self:maskInput(input), dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end