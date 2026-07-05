{
  description = "AI Swarm Intelligence — Evolutionary GPU Compute Sandbox";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in
      {
        devShells.default = pkgs.mkShell {
          name = "swarm-evolution-sandbox";

          buildInputs = with pkgs; [
            python3
            git
            gh
            bc
            opencode
            expect
            util-linux
            rustc
            cargo
            gcc
            pkg-config
            libiconv
          ];

          shellHook = ''
            echo "[swarm] Polyglot evolution sandbox ready — $(python3 --version)"
          '';
        };
      });
}
