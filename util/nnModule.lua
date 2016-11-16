--
-- Utilities for modules
--

function nn.Module:name(name)
    self._name = name
    return self
end

function nn.Module:findByName(name)
    if self._name == name then return self end
    if self.modules ~= nil then
        for i = 1, #self.modules do
            local module = self.modules[i]:findByName(name)
            if module ~= nil then return module end
        end
    end
end