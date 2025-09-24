return {
  {
    "folke/flash.nvim",
    event = "VeryLazy",
    ---@type Flash.Config
    opts = { jump = { autojump = true } },
      -- stylua: ignore
      keys = {
        { "s", mode = { "n", "x", "o" }, function() require("flash").jump() end, desc = "Flash" },
        { "S", mode = { "n", "x", "o" }, function() require("flash").treesitter() end, desc = "Flash Treesitter" },
        { "r", mode = "o", function() require("flash").remote() end, desc = "Remote Flash" },
        { "R", mode = { "o", "x" }, function() require("flash").treesitter_search() end, desc = "Treesitter Search" },
        { "<c-s>", mode = { "c" }, function() require("flash").toggle() end, desc = "Toggle Flash Search" },
      },
  },
  {
    "stevearc/oil.nvim",
    ---@module 'oil'
    ---@type oil.SetupOpts
    opts = {},
    -- Optional dependencies
    dependencies = { { "echasnovski/mini.icons", opts = {} } },
    -- dependencies = { "nvim-tree/nvim-web-devicons" }, -- use if you prefer nvim-web-devicons
    -- Lazy loading is not recommended because it is very tricky to make it work correctly in all situations.
    lazy = false,
  },
  {
    "sainnhe/sonokai",
    priority = 1000,
    config = function()
      -- Available values:   `'default'`, `'atlantis'`, `'andromeda'`, `'shusia'`, `'maia'`, `'espresso'`
      vim.g.sonokai_enable_italic = true
      vim.g.sonokai_style = "atlantis"
    end,
  },
  {
    "nvim-telescope/telescope.nvim",
    -- tag = "0.1.8",
    -- or                              , branch = '0.1.x',
    dependencies = { "nvim-lua/plenary.nvim" },
  },
  {
    "stevearc/conform.nvim",
    optional = true,
    opts = {
      formatters_by_ft = {
        -- ["python"] = { "ruff" },
        -- ["lua"] = { "stylua" },
        -- ["sh"] = { "shfmt" },
        python = { "ruff", "isort" },
        lua = { "stylua" },
        sh = { "shfmt" },
        toml = { "taplo" },
        md = { "mdformat" },
        fish = { "fish_indent" },
      },
    },
  },
}
