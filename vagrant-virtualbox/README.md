
# Vagrant VirtualBox install of a GUI bomberman_rl environment

Start by installing [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and [Vagrant](https://www.vagrantup.com/downloads.html) for your operating system.

* [VirtualBox](https://www.virtualbox.org/wiki/Downloads)
* [Vagrant](https://www.vagrantup.com/downloads.html)


After that install the vagrant virtual box guest additions plugin. This will ensure that your guest additions are always up-to-date when you start the vagrant box.

    vagrant plugin install vagrant-vbguest


Next change into the `vagrant-virtualbox` directory and download [Anaconda](https://www.anaconda.com/distribution/) via executing the `download.sh` script:

    cd $PATH_TO_BOMBERMAN_RL/vagrant-virtualbox
    ./download.sh


This will download the Anaconda installer into the local directory, so that if you would need to repeat the provisioning process the download does not need to happen again.


Now you're ready to provision the virtual box via executing the following command in the `vagrant-virtualbox` directory:

    vagrant up


You can go and grab a coffee. This will run for some time, depending on your internet download speed and hardware (expect something like 20 minutes).


After provisioning has finished you need to stop the VM and start it again so that it comes up in GUI mode with a display manager:

    vagrant halt
    vagrant up


Once the login appears **remember to first increase the VM window size** before you login! Otherwise you will end-up with a too small window to run the game.


The login and password is `vagrant` and `vagrant`. Once you're in the desktop environment use the `System Tools/LXTerminal` and change into the `bomberman_rl` directory and execute the following commands:

    cd bomberman_rl
    conda activate bomberman_rl
    python main.py


You should now see the `red` [agent_010_shred](https://github.com/cs224/bomberman_rl/tree/master/agent_code/agent_010_shred), the `yellow` and `green` instance of the [agent_011_shred](https://github.com/cs224/bomberman_rl/tree/master/agent_code/agent_011_shred) and the `blue` simple_agent. The *agent_010_shred* is the old version with a simpler model and the *agent_011_shred* is the new agent with a bit deeper model. The *simple_agent* is from the original task here: https://github.com/ukoethe/bomberman_rl.


If you would need to become root in the terminal you can use:

    sudo su


## Install description

The `Vagrantfile` is a `ruby` script that describes the fully automated provisioning process. You can inspect it to see how to set-up the environment for yourself on your own machine. The relevant parts are the following:

    sh /vagrant/Anaconda3-2019.03-Linux-x86_64.sh -b -p /home/vagrant/anaconda3

    echo "conda set-up path"
    export PATH="/home/vagrant/anaconda3/bin:$PATH"
    echo 'export PATH="/home/vagrant/anaconda3/bin:$PATH"' >> /home/vagrant/.bashrc

    conda update conda -y
    pip install --upgrade pip

    conda init bash

    __conda_setup="$('/home/vagrant/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    eval "$__conda_setup"

    git clone https://github.com/cs224/bomberman_rl
    cd bomberman_rl/agent_code/agent_011_shred/conda_environment
    ./create.sh
    conda activate bomberman_rl


## Clean-up

If you want to get rid-off the environment use:

    vagrant destroy -f


Use `vagrant box list` to see which "boxes" are still around and use `vagrant box remove NAME`. See the documentation for more details: https://www.vagrantup.com/docs/cli/box.html
