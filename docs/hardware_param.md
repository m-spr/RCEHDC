The elegance of this work lies in its reconfigurable architecture, which can be tailored to any problem of any size. Using the RCEHDC accelerator for different problems needs no additional hardware modifications. Customizations can be made simply by setting general parameters in the top module. All these paramiters automatically will be genreated for each module with the best configuration for minimize memory and hardware. The following section describes these parameters with their default values (for MNIST) and the variables for configuring them:



- **inbit**: `INTEGER := 8`
  - Defines the number of bits for input

- **dimension**: `INTEGER := 1000`
  - The HDC Dimension size

- **pruning**: `INTEGER := 336`
  - The number of efficient dimensions

- **featureSize**: `INTEGER := 784`
  - The number of elements in each input data

- **logfeature**: `INTEGER := 10`
  - LOG2(featureSize)

- **classes**: `INTEGER := 10`
  - Number of classes

- **classMemSize**: `INTEGER := 7`
  - The length of each segment of classHyper memories

- **confCompNum**: `INTEGER := 3`
  - Number of confComp modules in the comparator, calculated as ceiling(dimension/(2^classMemSize))

- **rsaZeropadding**: `INTEGER := 1`
  - The number of zero paddings for the sequential adder (RSA)

- **comparatorZeroPadding**: `INTEGER := 6`
  - The number of zero paddings for the multiplexer in comparators

- **logClasses**: `INTEGER := 4`
  - LOG2(classes)

- **IDreminder**: `INTEGER := 232`
  - Remainder value for ID-level

- **IDcoefficient**: `INTEGER := 3`
  - The coefficient of ID-level
