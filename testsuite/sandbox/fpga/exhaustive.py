# Here is a sample strategy file:
#{
#  "virtex6": {
#    "XST options": (
#      {"name": "my_xst1",  "xst" : " -opt_mode speed -opt_level 1"},
#      {"name": "my_xst2",  "xst" : " -opt_mode speed -opt_level 2"},
#      {"name": "my_xst3",  "xst" : " -opt_mode area  -opt_level 1"},
#      {"name": "my_xst4",  "xst" : " -opt_mode area  -opt_level 2"},
#    ),
#    "Map-Par options": (
#      {"name": "my_impl1", "map" : " -timing -ol high -xe n -global_opt on -retiming on ", "par" : " -ol high"},
#      {"name": "my_impl2", "map" : " -timing -ol high -xe n ", "par" : " -ol high"},
#    ),
#  }
#}


# This is a verbose enumeration of all options
#xst:
# -opt_mode (speed|area) //the first option is the default option
# -opt_level (1|2)
#
#map:
# -global_opt (off|speed|area|power)
# -ol (high|std)
# -xe (n|c)
# -timing
# -logic_opt (on|off)
# -cm (area|speed|balanced)
# -power (off|on|high|xe)
# -pr (off|i|o|b)
#
#par:
# -ol (high|std)
# -xe (n|c)
# -power (off|on)


# Here is the python code to generate an exhaustive strategy:
import itertools
xst1=(" -opt_mode ",["speed","area"])
xst2=(" -opt_level ",["1","2"])
xstA=tuple([{"name":"my_xst"+str(c), "xst":xst1[0]+o1+xst2[0]+o2}
for o1,i in zip(xst1[1],itertools.count()) for o2,c in zip(xst2[1],itertools.count(i*len(xst2[1])))])

map1=(" -global_opt ",["off","speed","area","power"])
map2=(" -ol ",["high","std"])
map3=(" -xe ",["n","c"])
map4=(" ",["","-timing"])
map5=(" -logic_opt ",["on","off"])
#map6=(" -cm ",["area","speed","balanced"])
map7=(" -power ",["off","on","high","xe"])
map8=(" -pr ",["off","i","o","b"])
mapA=[(map1[0]+o1+map2[0]+o2+map3[0]+o3+map4[0]+o4+map5[0]+o5+map7[0]+o7+map8[0]+o8)
for o1 in map1[1] for o2 in map2[1] for o3 in map3[1] for o4 in map4[1] for o5 in map5[1]
  for o7 in map7[1] for o8 in map8[1]]

par1=(" -ol ",["high","std"])
par2=(" -xe ",["n","c"])
par3=(" -power ",["off","on"])
parA=[(par1[0]+o1+par2[0]+o2+par3[0]+o3) for o1 in par1[1] for o2 in par2[1] for o3 in par3[1]]

MapParAll=tuple([{"name":"my_impl"+str(c),"map":mo, "par":po}
for mo,i in zip(mapA,itertools.count())
  for po,c in zip(parA,itertools.count(i*len(parA)))])

strategy={"virtex6":{"XST options":xstA, "Map-Par options": MapParAll}}
f=open("exhaustive.strategy", 'w')
f.write(str(strategy))
f.close()
