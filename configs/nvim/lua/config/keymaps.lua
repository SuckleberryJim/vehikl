-- Keymaps are automatically loaded on the VeryLazy event
-- Default keymaps that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/keymaps.lua
-- Add any additional keymaps here

vim.g.mapleader = " "
local map = vim.keymap.set

map("n", ";", ":")
map("n", "<leader>w", vim.cmd.w)
map("n", "<leader>q", vim.cmd.q)
map("n", "<leader>h", ":h ")

-- map("n", "<leader>e", vim.cmd.Ex)
-- map("n", "<leader>", "<c-w>")
map("n", ";", ":")
map("n", "<leader>;", ":!")

-- flash remaps
map("n", "r", "s")
map("n", "R", "S")
map("n", "C", "S")

-- buffer
map("n", "<leader>n", vim.cmd.bn)
map("n", "<leader>p", vim.cmd.bp)
map("n", "<leader>x", vim.cmd.bd)

-- terminal cmds
map("t", "<esc>", "<c-\\><c-n>")
map("n", "<leader>t", vim.cmd.terminal)

-- local netrw = {
--   netrw_banner = 0,
--   netrw_liststyle = 1,
--   netrw_sort_by = "modified",
--   netrw_sort_direction = "reverse",
-- }
--
-- for o, v in pairs(netrw) do
--   vim.g[o] = v
-- end

-- comment shortcut
map("v", "<c-/>", "gc")

-- vim.opt.formatoptions:remove('o', 'r', 'c')
vim.cmd([[autocmd BufEnter * set formatoptions-=cro]])

-- :h highlight-groups
local hl_groups = { "normal", "normalnc", "cursorline", "statusline", "statuslinenc", "title", "winbar", "winbarnc" }

for i = 1, #hl_groups do
  vim.api.nvim_set_hl(0, hl_groups[i], { bg = "none" })
end

-- vim.api.nvim_set_hl(0, "normal", { bg = "none" })
-- vim.api.nvim_set_hl(0, "normalnc", { bg = "none" })
-- vim.api.nvim_set_hl(0, "cursorline", { bg = "none" })
