#!/usr/bin/env python3
"""Print the verified exact stage whitelist; never invoke Git or expand a glob."""
import argparse
import json

import check


def render(whitelist, format):
    prefix = whitelist['project_prefix']
    if format == 'json':
        return json.dumps([prefix + '/' + name for name in whitelist['files']], indent=2) + '\n'
    check.require(format == 'tsv', 'explicit json or tsv format required')
    rows = ['group\tpath']
    for group, names in whitelist['groups'].items():
        rows.extend(group + '\t' + prefix + '/' + name for name in names)
    return '\n'.join(rows) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=('json', 'tsv'), required=True)
    args = parser.parse_args(argv)
    check.verify()
    print(render(check.load(check.HERE / 'STAGE_FILES.json'), args.format), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
