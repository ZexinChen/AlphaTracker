
# if install button is clicked, this script will be called

echo install start

#sleep 8
#exit 0


# conda env create -f environment.yml
conda create -n alphatracker python=3.8 -y

echo ". ~/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source  ~/.bashrc

conda activate alphatracker
# conda env update --file environment.yml

# Install pytorch
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

python setup.py build develop

# modify Makefile according to cuda version
cd ./Tracking/AlphaTracker/train_yolo/darknet/
make
cd ../../../../


conda activate alphatracker
cd ./Tracking/AlphaTracker/
python3 download.py


echo install over


