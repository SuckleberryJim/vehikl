# add user to sudoers file
# do not run this script as root
# su root
# enter pw
# usermod -aG sudo {username}

# system update
sudo apt update -y
sudo apt upgrade -y
sudo apt install curl git -y

# brew install script
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo >>~/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >>~/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

# pkg installs
brew install python neovim fish fastfetch bat eza ruff stylua shfmt taplo mdformat isort prettier uv tmux tree

# shell setup
fish
fish_add_path /home/linuxbrew/.linuxbrew/bin

# sudo echo >>/etc/shells
# sudo echo '/home/linuxbrew/.linuxbrew/bin/fish' >>/etc/shells
# sudo chsh -s /home/linuxbrew/.linuxbrew/bin/fish

# add fish shell to shells path
sudo vim /etc/shells

# neovim & fish
cp ./.config/.tmux.conf ~/
mkdir -p ~/.config/nvim
mkdir -p ~/.config/fish
cp ./.config/config.fish ~/.config/fish/
cp -r ./.config/nvim ~/.config/

brew list
fastfetch

fish_add_path /home/linuxbrew/.linuxbrew/bin

# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" |
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo systemctl start docker

# add user to docker group
sudo usermod -aG docker $USER
