rm -rf build dist
rm fot_planner.cpython-38-x86_64-linux-gnu.so
# replace with your path to site package if needed
#rm ~/anaconda3/envs/erdos/lib/python3.8/site-packages/fot_planner-1.0.0-py3.8-linux-x86_64.egg/fot_planner.cpython-38-x86_64-linux-gnu.so
rm -rf /usr/local/lib/python3.6/dist-packages/fot_planner-1.0.0-py3.6-linux-x86_64.egg/
./build.sh
# if you want to build and install locally 
# python setup.py build_ext --inplace 
# sudo python3 setup.py install
python3 setup.py install
