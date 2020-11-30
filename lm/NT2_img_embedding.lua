require 'nn'
require 'nngraph'

local img_embedding = {}

function img_embedding.img_embedding(hidden_size, fc_size, conv_size, conv_num, dropout)
	local inputs = {}
	local outputs = {}

	table.insert(inputs, nn.Identity()()) -- image feature 

  	local fc_feat = inputs[1]

    -- embed the fc7 feature -- dropout here? 
    local fc_feat_out = nn.ReLU()(nn.Linear(fc_size, hidden_size)(fc_feat))
    --if dropout > 0 then fc_feat_out = nn.Dropout(dropout)(fc_feat_out) end

  	-- local embed_feat_out = nn.Linear(hidden_size, hidden_size)(fc_feat_out)
 
    table.insert(outputs, fc_feat_out)
	-- table.insert(outputs, embed_feat_out)

  	return nn.gModule(inputs, outputs)

end

return img_embedding

