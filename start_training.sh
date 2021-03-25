# Select GPU
sudo nvidia-xconfig -a --cool-bits=28 --allow-empty-initial-configuration
#sudo nvidia-smi -i 0 -pm ENABLED
nvidia-settings -a "[gpu:0]/GPUFanControlState=1"

# Set GPU fan / core clock / memory clock / power limit
nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=65"
#nvidia-settings -a "[gpu:0]/GPUGraphicsClockOffset[3]=0"
#nvidia-settings -a "[gpu:0]/GPUMemoryTransferRateOffset[3]=0"
#sudo nvidia-smi -pl 175

# Start training
python3 -u main_informer.py

# Reset GPU fan
nvidia-settings -a "[fan:0]/GPUTargetFanSpeed=30"
