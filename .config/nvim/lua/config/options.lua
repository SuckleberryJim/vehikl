-- Options are automatically loaded before lazy.nvim startup
-- Default options that are always set: https://github.com/LazyVim/LazyVim/blob/main/lua/lazyvim/config/options.lua
-- Add any additional options here

local options = {
  ai = true,
  cb = "unnamedplus",
  ch = 1,
  -- cuc = true,
  cul = true,
  et = true,
  --
  list = false,
  nu = true,
  nuw = 1,
  rnu = true,
  -- omnifunc = ''
  -- ofu =
  so = 15,
  sw = 2,
  sc = true,
  swf = false,
  tgc = true,
  -- wrap = false,
  -- vbs = true,
  ts = 2,
  fdm = "expr",
  -- format options
  -- fo = "-cro",
  -- foldenable
  -- fen = true,
  fen = false,
  -- foldnestmax
  fdn = 1,
  -- signcolumn
  scl = "yes",
  -- searching
  hls = false,
  -- ignore case works for commands in command mode!!!
  ic = true,
  --
  -- column line to hl
  -- cc = "80",
}

options.foldexpr = "v:lua.vim.treesitter.foldexpr()"

for o, v in pairs(options) do
  vim.o[o] = v
end
