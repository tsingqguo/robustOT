#!/bin/sh

# CSA <-> pix2pix
if [ ! -d "libs/pix2pix" ]; then
    git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git libs/pix2pix
    cp libs/mods/pix2pix/models/* libs/pix2pix/models
    python scripts/install/rewrite_pix2pix_import.py
fi
