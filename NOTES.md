## Generator
- Series of tranpose convultional layers
- Output: Sigmoid activation function
- Input: image of 28x28x1 as the output is a greyscale image, z = latent dimension


-Batch normlization
    - method to make training nn's faster
    works by nromalising the output of the previous activation layer by subtracting the batch meand and dividing by the batch std


- Convultional transpose layer
    - typically used for upsampling
        - generate an output feature map that has a spatial dimension greater than the input feature map
    - like a normal convultion layer, but padding and stride are defined to be
        - padding = k-p-1
        - stride =1


## Discriminator
- Series of convultional layers that are normlised and activated with Leaky ReLU
- Output: Value between 0 and 1