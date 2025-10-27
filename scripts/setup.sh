# # This script sets up the conda environment for the project
# conda create -n lemexp
# conda activate lemexp

# # Install the required packages
# conda install python=3.12.8

# # Install the required packages
# pip install -r requirements.txt

# Install Isabelle 2024
wget https://isabelle.in.tum.de/dist/Isabelle2025_linux.tar.gz 
tar -xzf Isabelle2024_linux.tar.gz
rm Isabelle2024_linux.tar.gz

# Install AFP
wget https://www.isa-afp.org/release/afp-current.tar.gz
tar -xzf afp-current.tar.gz
rm afp-current.tar.gz

alias isabelle="Isabelle2024/bin/isabelle"
isabelle build -b -D Isabelle2024/src/HOL -j 4

# Register AFP
isabelle components -u afp/thys