# Select GPU
sudo nvidia-xconfig -a --cool-bits=28 --allow-empty-initial-configuration
nvidia-settings -a "[gpu:0]/GPUFanControlState=1"

# Set GPU fan / core clock / memory clock / power limit
nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=75"
nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[3]=-200"
nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=-200"
sudo nvidia-smi -pl 160

# Start the Ethminer
python3 -u main_informer.py --model informer --data custom
