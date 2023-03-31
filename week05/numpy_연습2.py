import numpy as np
a=np.arange(16)

b=a.reshape(1, 16)

print(a)
print(b)

import numpy as np
a=np.arange(16)
b=a.reshape(2, -1)



print(a)
print(b)

import numpy as np
a=np.arange(16)
b=a.reshape(1, -1, 2)

print(a)
print(b)

import numpy as np
a=np.arange(12)
b=a.reshape(4, -1, 1)

print(a)
print(b)

import numpy as np
a=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
b=a[1, 3].reshape(1)
print(b)


