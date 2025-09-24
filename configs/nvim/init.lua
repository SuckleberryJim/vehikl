-- bootstrap lazy.nvim, LazyVim and your plugins
require("config.lazy")

local map = vim.keymap.set

-- telescope
-- local builtin = require("telescope.builtin")
-- vim.keymap.set("n", "<leader>f", builtin.find_files, { desc = "Telescope find files" })
-- vim.keymap.set("n", "<leader>fg", builtin.live_grep, { desc = "Telescope live grep" })
-- vim.keymap.set("n", "<leader>fb", builtin.buffers, { desc = "Telescope buffers" })
-- vim.keymap.set("n", "<leader>fh", builtin.help_tags, { desc = "Telescope help tags" })
local telescope = require("telescope.builtin")
map("n", "<leader>f", telescope.find_files, { desc = "Telescope find files" })

-- conform
require("conform").setup({
  formatters_by_ft = {
    lua = { "stylua" },
    python = { "isort", "ruff_format" }, -- or black
    markdown = { "mdformat" },
    sh = { "shfmt" },
    json = { "prettier" },
    toml = { "taplo" },
    fish = { "fish_indent" },
  },
})

-- map("n", "<leader>b", ":NvimTreeToggle<cr>")

--oil
require("oil").setup({
  view_options = { show_hidden = true },
})
map("n", "-", "<cmd>Oil<cr>")

-- terminal
map("t", "<esc>", "<c-\\><c-n>")
-- ft and fT toggle the terminal btw

vim.cmd.colorscheme("catppuccin-macchiato")

vim.opt.wrap = true
vim.opt.breakindent = true
vim.opt.linebreak = true
