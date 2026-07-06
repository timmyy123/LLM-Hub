# THIS REPO WILL SOON BE ARCHIVED AND NO LONGER MAINTAINED

# Please use the Swift package directly from [whisper.cpp](https://github.com/ggerganov/whisper.cpp)

---

# whisper.spm

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) package for the Swift Package Manager

## Using as a package in your project

In XCode: File -> Add Packages

Enter package URL: `https://github.com/ggerganov/whisper.spm`

⚠️ Important: Set Dependency Rule to 'branch' -> 'master' otherwise you get an [Error with unsafe build flags](https://github.com/ggerganov/whisper.spm/issues/4)

<img width="1091" alt="image" src="https://user-images.githubusercontent.com/1991296/200189694-aed421ae-6fd7-4b17-8211-e43040c32e97.png">

## Build package from command line

```bash
git clone https://github.com/ggeragnov/whisper.spm
cd whisper.spm

# if building standalone
make build

# if building as a submodule for whisper.cpp
make build-submodule

# run tests
.build/debug/test-objc
.build/debug/test-swift
```
