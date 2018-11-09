# Levenshtein
pip install python-Levenshtein

# Warp-ctc
git clone https://github.com/SeanNaren/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make

cd ..
cd pytorch_binding
python setup.py install

# # dlopen error
# cd ../pytorch_binding
# python setup.py install
# cd ../build
# cp libwarpctc.dylib /Users/$WHOAMI/anaconda3/lib

# ctcdecode
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode
pip install .