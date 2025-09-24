# add user to sudoers file

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
brew install python neovim fish docker fastfetch bat eza ruff stylua shfmt taplo mdformat isort prettier uv tmux tree

# shell setup
fish
fish_add_path /home/linuxbrew/.linuxbrew/bin
sudo echo >>/etc/shells
sudo echo '/home/linuxbrew/.linuxbrew/bin/fish' >>/etc/shells
chsh -s /home/linuxbrew/.linuxbrew/bin/fish
# neovim & fish
cp -r ./.config ~/.config
cp ./.config/.tmux.conf ~/
cp ./.config/config.fish ~/.config/fish/

# add user to docker group

brew list
fastfetch
