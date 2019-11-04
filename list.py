with open("list.txt","w") as f:
    # for c in ["000", "001", "008", "027", "064", "125"]:
    for c in range(7):
        f.write(str(c) + ".mp4 \n")
    f.close()
