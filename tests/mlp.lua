require 'nngraph'

function t3()
   mlp=nn.Sequential();       --Create a network that takes a Tensor as input
   mlp:add(nn.SplitTable(2))
   c=nn.ParallelTable()      --The two Tensors go through two different Linear
   c:add(nn.Linear(10,3))	   --Layers in Parallel
   c:add(nn.Linear(10,7))
   mlp:add(c)                 --Outputing a table with 2 elements
   p=nn.ParallelTable()      --These tables go through two more linear layers
   p:add(nn.Linear(3,2))	   -- separately.
   p:add(nn.Linear(7,1))
   mlp:add(p)
   mlp:add(nn.JoinTable(1))   --Finally, the tables are joined together and output.

   pred=mlp:forward(torch.randn(10,2))
   print("prediction: ")
   print(pred)

   for i=1,25 do             -- A few steps of training such a network..
      x=torch.ones(10,2);
      y=torch.Tensor(3); y:copy(x:select(2,1,1):narrow(1,1,3))
      pred=mlp:forward(x)

      criterion= nn.MSECriterion()
      local err=criterion:forward(pred,y)
      local gradCriterion = criterion:backward(pred,y);
    --   print('x and y: ')
    --   print(x,y)
      mlp:zeroGradParameters();
      mlp:backward(x, gradCriterion);
      mlp:updateParameters(0.05);

      print(err)
   end
end

t3()
