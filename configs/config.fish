if status is-interactive
    # Commands to run in interactive sessions can go here
end

# we should break up aliases, functions, vars, etc. into various files

# add brew path
fish_add_path /home/linuxbrew/.linuxbrew/bin/fish

alias l='eza -lAh'
alias ls='eza -lh'
alias py=python3
alias k=clear

alias t=tmux
alias tr=tree
alias v=nvim

alias cp='cp -v'
alias rm='rm -v'
alias mv='mv -v'

alias mkdir='mkdir -v'
# alias t=touch

alias a=alias
alias m=man
alias f=fastfetch

alias '..'='cd ../'
alias '...'='cd ../../'
alias '....'='cd ../../../'

lias d=docker
alias db='docker build'
alias dr='docker run'
alias dc='docker compose'
alias dcb='docker compose build'
alias dcu='docker compose up'
alias dcd='docker compose down'

bind escape forward-char
bind ctrl-o 'nvim .'
bind alt-o 'nvim .'
# fish does not support cmd
# bind cmd-o 'nvim .'

function e
    nvim ~/.config/fish/config.fish
    source ~/.config/fish/config.fish
end

# choose and set a fish prompt | pretty good defaults to choose from
# fish_config prompt list
fish_config prompt choose minimalist

# function sync
#   git add .
#   git commit -m

function u
    # brew analytics off
    brew update
    brew upgrade
    brew cleanup
    brew autoremove
    echo '~update complete~'
    fastfetch
end

function sync
    echo $argv
    echo $(count $argv)
    echo $argv[1]

    git add .
    git commit -m $argv[1]
    git push -uv origin main
    git status
end

function r
    npm run dev -- --open
end

# Added by Windsurf
fish_add_path /Users/atriox/.codeium/windsurf/bin

alias ur='uv run'
alias uvenv='source .venv/bin/activate.fish'
alias uvi='uv init . --bare'
