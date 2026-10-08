import subprocess
import sys
import os
import json


def main(root):
    execroot = (
        subprocess.check_output(["bazel", "info", "execution_root"])
        .decode("utf-8")
        .strip()
    )
    aquery = json.loads(
        subprocess.check_output(
            [
                "bazel",
                "aquery",
                "--compilation_mode=dbg",
                "mnemonic(SwiftCompile, //...)",
                "--output=jsonproto",
                "--include_artifacts=false",
            ]
        )
    )
    actions = aquery["actions"]

    def command(action):
        arguments = list(
            filter(
                lambda x: "worker/worker" not in x and "-Xwrapped-swift" not in x,
                action["arguments"],
            )
        )
        arguments[-1] = os.path.join(root, arguments[-1])
        return {"directory": execroot, "arguments": arguments, "file": arguments[-1]}

    compile_commands = map(command, actions)
    with open("compile_commands.json", "w+") as f:
        json.dump(list(compile_commands), f, sort_keys=True, indent=2)


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    main(sys.argv[1])
