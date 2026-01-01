#!/usr/bin/env bash
#run this in windows not wsl bc for some reason my apt can't find protoc>=3.19
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/net.proto
protoc --proto_path=libs/lczero-common --python_out=tf libs/lczero-common/proto/chunk.proto
touch tf/proto/__init__.py
