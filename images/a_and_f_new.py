import numpy as np
import matplotlib.pyplot as plt
import math
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Applications.PVSnapshot import PVSnapshot
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam . Applications . Runner import Runner
import shutil
import os
from paraview.simple import *
import glob

data= [{"target": 29.39876365544583, "params": {"a": 0.001, "f": 2.0}, "datetime": {"datetime": "2022-09-02 20:03:13", "elapsed": 0.0, "delta": 0.0}},
        {"target": 12.335942416453731, "params": {"a": 0.00275, "f": 3.5}, "datetime": {"datetime": "2022-09-02 20:50:06", "elapsed": 2812.824329, "delta": 2812.824329}},
        {"target": 5.468665085872916, "params": {"a": 0.0045000000000000005, "f": 6.5}, "datetime": {"datetime": "2022-09-02 22:28:22", "elapsed": 8708.539412, "delta": 5895.715083}},
        {"target": 1.6265610535790858, "params": {"a": 0.00625, "f": 8.0}, "datetime": {"datetime": "2022-09-03 01:36:45", "elapsed": 20011.84393, "delta": 11303.304518}},
        {"target": 4.26480023790481, "params": {"a": 0.008, "f": 5.0}, "datetime": {"datetime": "2022-09-03 04:02:58", "elapsed": 28784.488189, "delta": 8772.644259}},
        {"target": 6.054685758540129, "params": {"a": 0.00799999999999991, "f": 2.4639638876062153}, "datetime": {"datetime": "2022-09-03 05:04:19", "elapsed": 32465.773341, "delta": 3681.285152}},
        {"target": 6.226157349395247, "params": {"a": 0.007099378087402528, "f": 3.9827301128327255}, "datetime": {"datetime": "2022-09-03 06:45:37", "elapsed": 38543.786318, "delta": 6078.012977}},
        {"target": 5.0382342661193995, "params": {"a": 0.004949240117049483, "f": 3.821730679926052}, "datetime": {"datetime": "2022-09-03 07:41:13", "elapsed": 41879.184886, "delta": 3335.398568}},
        {"target": 14.197307724325118, "params": {"a": 0.002546455802954157, "f": 3.072565315178941}, "datetime": {"datetime": "2022-09-03 08:29:15", "elapsed": 44761.424603, "delta": 2882.239717}},
        {"target": 29.082662924950508, "params": {"a": 0.001, "f": 5.763853268759136}, "datetime": {"datetime": "2022-09-03 11:19:27", "elapsed": 54973.443451, "delta": 10212.018848}},
        {"target": 5.39526406503317, "params": {"a": 0.008, "f": 5.491722919721239}, "datetime": {"datetime": "2022-09-03 14:02:06", "elapsed": 64732.816225, "delta": 9759.372774}},
        {"target": 9.72488775379098, "params": {"a": 0.005227350561162486, "f": 2.122967182751228}, "datetime": {"datetime": "2022-09-03 14:33:45", "elapsed": 66631.207039, "delta": 1898.390814}},
        {"target": 24.529547708586303, "params": {"a": 0.001196685324209766, "f": 6.101415900285111}, "datetime": {"datetime": "2022-09-03 16:11:02", "elapsed": 72468.739734, "delta": 5837.532695}},
        {"target": 10.972635389541201, "params": {"a": 0.0035629290595777436, "f": 3.0832466677620696}, "datetime": {"datetime": "2022-09-03 16:46:48", "elapsed": 74614.969519, "delta": 2146.229785}},
        {"target": 24.33147110824823, "params": {"a": 0.001, "f": 5.810966042990315}, "datetime": {"datetime": "2022-09-03 19:36:51", "elapsed": 84817.902941, "delta": 10202.933422}},
        {"target": 16.033324708396666, "params": {"a": 0.0020372848213604747, "f": 7.788347421024929}, "datetime": {"datetime": "2022-09-03 20:36:03", "elapsed": 88369.405211, "delta": 3551.50227}},
        {"target": 1.6681117028562111, "params": {"a": 0.006314454171573853, "f": 5.692409802378555}, "datetime": {"datetime": "2022-09-03 22:38:14", "elapsed": 95701.028248, "delta": 7331.623037}},
        {"target": 6.437185507060833, "params": {"a": 0.0079860655220965, "f": 2.0435532688972544}, "datetime": {"datetime": "2022-09-03 23:20:37", "elapsed": 98243.788447, "delta": 2542.760199}},
        {"target": 24.840876771839216, "params": {"a": 0.0016755692687346595, "f": 6.063047634641915}, "datetime": {"datetime": "2022-09-04 00:03:39", "elapsed": 100826.08311, "delta": 2582.294663}},
        {"target": 5.628608634180504, "params": {"a": 0.00704711797919158, "f": 6.011679933494367}, "datetime": {"datetime": "2022-09-04 02:38:19", "elapsed": 110105.585577, "delta": 9279.502467}},
        {"target": 6.448794053445739, "params": {"a": 0.008, "f": 6.147834374965284}, "datetime": {"datetime": "2022-09-04 05:46:45", "elapsed": 121411.225549, "delta": 11305.639972}},
        {"target": 5.872045329921501, "params": {"a": 0.00799999999981321, "f": 5.852681280163261}, "datetime": {"datetime": "2022-09-04 08:44:08", "elapsed": 132054.494722, "delta": 10643.269173}},
        {"target": 7.632040285100563, "params": {"a": 0.004105769861241481, "f": 6.111355884080363}, "datetime": {"datetime": "2022-09-04 10:01:48", "elapsed": 136715.134794, "delta": 4660.640072}},
        {"target": 2.780091033402825, "params": {"a": 0.005454163207772318, "f": 7.471680266844707}, "datetime": {"datetime": "2022-09-04 12:24:42", "elapsed": 145288.651321, "delta": 8573.516527}},
        {"target": 6.980841107531957, "params": {"a": 0.00773558473041578, "f": 2.4277204638571908}, "datetime": {"datetime": "2022-09-04 13:24:29", "elapsed": 148875.913236, "delta": 3587.261915}},
        {"target": 6.709545932647131, "params": {"a": 0.00760353446276924, "f": 4.683546090870397}, "datetime": {"datetime": "2022-09-04 15:33:38", "elapsed": 156624.907928, "delta": 7748.994692}},
        {"target": 6.577878070607258, "params": {"a": 0.0075535200974529895, "f": 5.593644881360314}, "datetime": {"datetime": "2022-09-07 20:09:00", "elapsed": 166275.007359, "delta": 9650.099431}},
        #{"target": 27.826804697763258, "params": {"a": 0.001, "f": 5.0}, "datetime": {"datetime": "2022-09-07 20:09:00", "elapsed": 0.0, "delta": 0.0}},
        {"target": 10.274254617254757, "params": {"a": 0.00275, "f": 6.5}, "datetime": {"datetime": "2022-09-07 21:10:12", "elapsed": 3671.862175, "delta": 3671.862175}},
        {"target": 9.448543606360166, "params": {"a": 0.0045000000000000005, "f": 3.5}, "datetime": {"datetime": "2022-09-07 21:59:59", "elapsed": 6658.941295, "delta": 2987.07912}},
        {"target": 8.136147161727793, "params": {"a": 0.00625, "f": 2.0}, "datetime": {"datetime": "2022-09-07 22:38:16", "elapsed": 8955.89551, "delta": 2296.954215}},
        {"target": 8.291533603464622, "params": {"a": 0.008, "f": 8.0}, "datetime": {"datetime": "2022-09-08 02:57:51", "elapsed": 24530.71247, "delta": 15574.81696}},
        {"target": 9.048681112188962, "params": {"a": 0.003551673148276966, "f": 5.765081100869011}, "datetime": {"datetime": "2022-09-08 04:02:14", "elapsed": 28393.727477, "delta": 3863.015007}},
        {"target": 19.16414243865882, "params": {"a": 0.0023122032725472323, "f": 4.9985195746467355}, "datetime": {"datetime": "2022-09-08 04:45:01", "elapsed": 30961.431552, "delta": 2567.704075}},
        {"target": 15.148700461453931, "params": {"a": 0.0027611422073158106, "f": 2.0018168576442177}, "datetime": {"datetime": "2022-09-08 05:58:49", "elapsed": 35389.212078, "delta": 4427.780526}},
        {"target": 26.72129341286191, "params": {"a": 0.0017893207819753934, "f": 5.7615589454319505}, "datetime": {"datetime": "2022-09-08 06:43:37", "elapsed": 38077.602759, "delta": 2688.390681}},
        {"target": 22.55818157159876, "params": {"a": 0.0015778773961585535, "f": 5.003504331601452}, "datetime": {"datetime": "2022-09-08 07:29:48", "elapsed": 40847.728091, "delta": 2770.125332}},
        {"target": 21.467714434509126, "params": {"a": 0.0021979149229268647, "f": 5.00354411308975}, "datetime": {"datetime": "2022-09-08 08:11:16", "elapsed": 43336.544113, "delta": 2488.816022}}
]

freq= []
N= []
dt= []
amp= []
for i in range(1,len(data)):
    p= data[i]['params']
    f= round(p['f'],2)
    freq.append(f)
    a= round(p['a'],4)
    amp.append(a)
    t= round(data[i]['target'],2)
    N.append(t)
    d= data[i-1]['datetime']['datetime']
    dt.append(d)

sub_dt= dt[1:2]

for m in range(len(sub_dt)):
#test_num= get_value(amp[m],freq[m],N[m])
    folder_name= dt[m].replace(' ','_').replace('-','_').replace(':','_')
    print(folder_name)
    #os.mkdir(folder_name)

    file_path= "~/a_and_f/"
    expanded_file_path = os.path.expanduser(file_path)
    new_file_path= os.path.join(expanded_file_path, folder_name)

    png_files = glob.glob(os.path.join(new_file_path, '*streamlines_3.png'))
    
    shutil.copy2(os.path.join(expanded_file_path, 'streamlines_3.pvsm'), folder_name)

    # Newcase=folder_name
    # shutil.copyfile('streamlines_3.pvsm',os.path.join(folder_name,'streamlines_3.pvsm'))