if status is-interactive
    # Commands to run in interactive sessions can go here
end

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

alias d=docker
alias dcps='docker compose'

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
fish_config prompt choose disco # minimalist

# function sync
#   git add .
#   git commit -m

function u
    brew analytics off
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
    git push -u origin main
    git status
end

function r
    npm run dev -- --open
end

function uvr
    uv run $argv[1]
end

# run tmux on terminal open
# tmux
# run fastfetch on terminal open
# fastfetch

function uvi
    uv init --bare
end

# Added by Windsurf
fish_add_path /Users/atriox/.codeium/windsurf/bin
