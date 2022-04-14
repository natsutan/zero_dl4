import sys

sys.path.append("../common")
from World import GridWorld


def main():
    env = GridWorld()
    print(env.height)
    print(env.width)
    print(env.shape)

    for a in env.actions() :
        print(a)

    print("===")
    for s in env.states():
        print(s)


if __name__ == '__main__':
    main()
