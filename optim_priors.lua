function doStuff_temp(Models,criterion,Batch,coef)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, Model, Model2, State1, State2

   local batchSize = Batch:size(2)

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
   end

   Model=Models.Model1
   Model2=Models.Model2
   State1=Model:forward(im1)
   State2=Model2:forward(im2)

   assert(batchSize==State1:size(1), "Batch Size changed during 'forward method, maybe a nn.view is done badly ...")

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   loss=criterion:forward({State2,State1})
   GradOutputs=criterion:backward({State2,State1})

   -- calculer les gradients pour les deux images
   Model:backward(im1,coef*GradOutputs[2])
   Model2:backward(im2,coef*GradOutputs[1])
   return loss, coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Caus(Models,criterion,Batch,coef, action1, action2)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, Model, Model2, State1, State2

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
   end

   Model=Models.Model1
   Model2=Models.Model2

   State1=Model:forward(im1)
   State2=Model2:forward(im2)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2})
   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2}, torch.ones(1))

   -- compute the gradients for the two images
   if USE_CONTINUOUS then
       continuous_factor_term = get_continuous_action_factor_term(action1, action2)
       Model:backward(im1, continuous_factor_term * coef*GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2, continuous_factor_term * coef*GradOutputs[2]/Batch[1]:size(1))
   else
       Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
   end
   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Prop(Models,criterion,Batch, coef, action1, action2)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State1=Model:forward(im1)
   State2=Model2:forward(im2)
   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})

   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4},torch.ones(1))

   -- compute the gradients for the two images
   if USE_CONTINUOUS then
       continuous_factor_term = get_continuous_action_factor_term(action1, action2)
       Model:backward(im1, continuous_factor_term * coef *GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2, continuous_factor_term * coef *GradOutputs[2]/Batch[1]:size(1))
       Model3:backward(im3, continuous_factor_term * coef *GradOutputs[3]/Batch[1]:size(1))
       Model4:backward(im4, continuous_factor_term * coef *GradOutputs[4]/Batch[1]:size(1))
   else
       Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
       Model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
       Model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))
   end

   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

function doStuff_Rep(Models,criterion,Batch, coef, action1, action2)
    print('actions')
    print(action1)
   -- Returns the loss and the gradient
   local coef= coef or 1
   local im1, im2, im3, im4, Model, Model2, Model3, Model4, State1, State2, State3, State4

   if USE_CUDA then
      im1=Batch[1]:cuda()
      im2=Batch[2]:cuda()
      im3=Batch[3]:cuda()
      im4=Batch[4]:cuda()
   else
      im1=Batch[1]
      im2=Batch[2]
      im3=Batch[3]
      im4=Batch[4]
   end

   Model=Models.Model1
   Model2=Models.Model2
   Model3=Models.Model3
   Model4=Models.Model4

   State1=Model:forward(im1)--added for image 1 and 2
   State2=Model2:forward(im2)
   State3=Model3:forward(im3)
   State4=Model4:forward(im4)

   if USE_CUDA then
      criterion=criterion:cuda()
   end
   output=criterion:forward({State1, State2, State3, State4})

   --we backward with a starting gradient initialized at 1
   GradOutputs=criterion:backward({State1, State2, State3, State4}, torch.ones(1))

   -- compute the gradients for the two images
   if USE_CONTINUOUS then
       continuous_factor_term = get_continuous_action_factor_term(action1, action2)
       Model:backward(im1,continuous_factor_term * coef*GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2,continuous_factor_term * coef*GradOutputs[2]/Batch[1]:size(1))
       Model3:backward(im3,continuous_factor_term * coef*GradOutputs[3]/Batch[1]:size(1))
       Model4:backward(im4,continuous_factor_term * coef*GradOutputs[4]/Batch[1]:size(1))
   else
       Model:backward(im1,coef*GradOutputs[1]/Batch[1]:size(1))
       Model2:backward(im2,coef*GradOutputs[2]/Batch[1]:size(1))
       Model3:backward(im3,coef*GradOutputs[3]/Batch[1]:size(1))
       Model4:backward(im4,coef*GradOutputs[4]/Batch[1]:size(1))
   end
   return output:mean(), coef*GradOutputs[1]:cmul(GradOutputs[1]):mean()
end

--------------------------------------------------------------
----CONTINUOUS ACTION PRIORS VERSION
function actions_distance(action1, action2)
  -- Returns a double indicating the Euclidean distance among actions
  local distance = 0
  --for each dim, check that the magnitude of the action is close
  for dim=1, DIMENSION_IN do
     distance = distance + (math.pow(arrondit(action1[dim]) - arrondit(action2[dim]), 2))
  end
  return math.sqrt(distance)
end

function get_continuous_action_factor_term(action1, action2)
  -- This methods avoids having to check for actions that are close enough or
  -- far away enough for the priors constraints by multiplying by a continuous
  -- factor  based on sigma GAUSSIAN_SIGMA
  return math.exp((-1 * actions_distance(action1, action2))/GAUSSIAN_SIGMA)
end
