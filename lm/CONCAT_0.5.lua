require 'nn'
require 'nngraph'

local CONCAT = {}

function CONCAT.concat(rnn_size, output_size, dropout)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  local outputs = {}
  -- 给出符号序列的索引
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols
  table.insert(inputs, nn.Identity()())
  local xf = inputs[1]
  local proj_f = nn.MulConstant(0.5,false)(xf)

  local xb = inputs[2]
  local proj_b = nn.MulConstant(0.5,false)(xb)

  local proj = nn.CAddTable()({proj_f, proj_b})
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return CONCAT

