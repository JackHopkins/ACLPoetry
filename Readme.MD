#Installation
~~~
sudo apt-get -y install libhdf5-dev
sudo apt install -y virtualenv

virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment

git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;

#This command might not be needed
./install.sh

cd ..
git clone --recursive https://github.com/hughperkins/distro -b distro-cl ~/torch-cl
cd ~/torch-cl; bash install-deps
./install.sh

source ~/.bashrc

sudo apt-get install -y libhdf5-serial-dev hdf5-tools

luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson
luarocks install hdf5
#luarocks install cutorch

source ~/torch-cl/install/bin/torch-activate

sudo apt-get -y install libboost-all-dev

cd ~/
git clone --recursive https://github.com/graehl/carmel.git ~/graehl

cd graehl/carmel; sudo make -j 4 install BOOST_SUFFIX= INSTALL_PREFIX=/usr/local

git clone --recursive https://github.com/JackHopkins/ACLPoetry.git ~/ACLPoetry
~~~