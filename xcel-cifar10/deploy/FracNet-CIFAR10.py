
# coding: utf-8

# In[1]:


import sys, os, time
import numpy as np

from pynq import Xlnk
from pynq import Overlay
import pynq

xlnk = Xlnk()
xlnk.xlnk_reset()


# In[2]:


overlay = Overlay("./FracNet_T_0.bit")
# overlay?
FracNet = overlay.FracNet_T_0
# timer = overlay.axi_timer_0


# In[3]:


FracNet.register_map


# In[4]:


# timer.register_map


# In[5]:


bus512 = 'B,'*63 + 'B'
dt_512 = np.dtype(bus512)

bus256 = 'B,'*31 + 'B'
dt_256 = np.dtype(bus256)

image_thermo = xlnk.cma_array(shape=(3,32,32), dtype=np.uint64)
result = xlnk.cma_array(shape=(10), dtype=np.float32)


# In[6]:


import numpy as np
images = np.load('conv1_input_uint64.npy')


# In[7]:



num_tests = 1000
with open('labels.bin', 'rb') as f:
    content = f.read()
print(len(content))

labels = np.ndarray((num_tests,))
for i in range(num_tests):
    labels[i] = content[i]


# In[8]:


FracNet.register_map.image_V   = image_thermo.physical_address
FracNet.register_map.output_r  = result.physical_address
FracNet.register_map


# In[13]:


from time import perf_counter
idle = FracNet.register_map.CTRL.AP_IDLE
FracNet.register_map.CTRL.AP_START = 0

t = 0
correct = 0
for i in range(num_tests):
    np.copyto(image_thermo, images[i])
    idle = 0
    FracNet.register_map.CTRL.AP_START = 1
    
    ts = perf_counter()
    while idle == 0:
        idle = FracNet.register_map.CTRL.AP_IDLE
    tt = perf_counter()
    
    t += tt - ts
    
    pred = np.argmax(result)
    if pred == labels[i]:
        correct += 1
    
print('Latency: %.4f ms'%(t/num_tests*1000))
print('Throughput: %.4f fps'%(1/(t/num_tests)))
print('Accuracy: %.1f%%'%(correct/num_tests*100))

