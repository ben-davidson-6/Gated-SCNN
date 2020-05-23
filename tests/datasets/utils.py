import os


def list_files(startpath):
    """
    print structure of a directory
    :param startpath str directory to display contents of:
    :return:
    """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        if len(files) > 100:
            print('{}#{}files'.format(subindent, len(files)))
        else:
            for f in files:
                print('{}{}'.format(subindent, f))


