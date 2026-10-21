#!/usr/bin/env python3
# The safe entry points in rust/ against vk.xml: `&mut` only where the app synchronizes the
# parameter (externsync="true") or for an output pointer, a plain reference never where the
# parameter may be null, and an array's length is the last u32 before it (entrypoint.rs's Param).
# The struct accessors of members! name the member's type and length and never a member vk.xml
# leaves unvalidated.
import glob
import re
import sys
import xml.etree.ElementTree as ET


def u32_types(root):
    u32 = {'uint32_t', 'VkBool32', 'VkSampleMask', 'VkFlags'}
    for t in root.iter('type'):
        if t.get('category') == 'enum':
            u32.add(t.get('name'))
        elif t.get('category') == 'bitmask' and t.findtext('type') == 'VkFlags':
            u32.add(t.findtext('name') or t.get('name'))
    return u32


def commands(xml):
    root = ET.parse(xml).getroot()
    u32 = u32_types(root)
    cmds, aliases = {}, {}
    for c in root.iter('command'):
        if c.get('alias'):
            aliases[c.get('name')] = c.get('alias')
            continue
        if c.get('api', 'vulkan') == 'vulkansc' or c.find('proto') is None:
            continue
        params = []
        for p in c.findall('param'):
            if p.get('api', 'vulkan') == 'vulkansc':
                continue
            text = ''.join(p.itertext())
            params.append({
                'name': p.find('name').text,
                'pointer': '*' in text,
                'const': text.lstrip().startswith('const'),
                'externsync': p.get('externsync', ''),
                'optional': p.get('optional', 'false').split(',')[0] == 'true',
                'type': p.findtext('type'),
                'u32': '*' not in text and p.findtext('type') in u32,
                'len': p.get('len'),
            })
        cmds[c.find('proto/name').text] = params
    for a, target in aliases.items():
        if target in cmds:
            cmds[a] = cmds[target]
    return cmds


def structs(root):
    out = {}
    for t in root.iter('type'):
        if t.get('category') in ('struct', 'union') and t.get('api', 'vulkan') != 'vulkansc':
            out[t.get('name')] = {m.findtext('name'): m for m in t.findall('member') if m.get('api', 'vulkan') != 'vulkansc'}
    return out


def check_members(paths, xml):
    errors, found = [], 0
    sts = structs(ET.parse(xml).getroot())
    for path in paths:
        for blk in re.finditer(r'trait \w+ for (\w+) \{([^}]*)\}', open(path).read()):
            found += 1
            s = sts.get(blk.group(1))
            for m in re.finditer(r'(\w+): (\w+)(?:\[(\w+)\])? -> (\w+)', blk.group(2)):
                _, field, n, ty = m.groups()
                where = f'{path}: {blk.group(1)}.{field}'
                mem = s.get(field) if s else None
                if mem is None:
                    errors.append(f'{where} is not in vk.xml')
                elif mem.get('noautovalidity') == 'true':
                    errors.append(f'{where}: vk.xml does not validate it')
                elif '*' not in ''.join(mem.itertext()) or mem.findtext('type') != ty:
                    errors.append(f'{where}: vk.xml has {"".join(mem.itertext()).strip()}')
                elif mem.get('len') != n:
                    errors.append(f'{where}: its length is {mem.get("len")}, not {n}')
    return errors if found else ['no members! accessors found']


def signatures(paths):
    for path in paths:
        for m in re.finditer(r'^\s*fn borgvk_(\w+)\(([^)]*)\)', open(path).read(), re.M):
            params = [p.split(':', 1)[1].strip() for p in m.group(2).split(',') if ':' in p]
            yield path, 'vk' + m.group(1), params


def check(rust, p, last_u32):
    if re.match(r'&(mut )?\[', rust):
        return None if p['len'] == last_u32 else f'its length is {p["len"]}, the last u32 before it {last_u32}'
    mut = re.match(r'(Option<)?&mut ', rust)
    ref = re.match(r'(Option<)?&', rust)
    if mut and p['pointer'] and p['const']:
        return '&mut of an input'
    if mut and not p['pointer'] and p['externsync'] != 'true':
        return '&mut of a handle the app does not synchronize'
    if ref and not rust.startswith('Option<') and p['optional']:
        return 'a reference where the parameter may be null'
    return None


def main(xml, rust_dir):
    cmds = commands(xml)
    errors = []
    paths = sorted(glob.glob(rust_dir + '/*.rs'))
    sigs = list(signatures(paths))
    errors += check_members(paths, xml)
    if not sigs:
        errors.append(f'no entry points found in {rust_dir}')
    for path, name, params in sigs:
        xparams = cmds.get(name)
        if xparams is None:
            errors.append(f'{path}: {name} is not in vk.xml')
        elif len(params) != len(xparams):
            errors.append(f'{path}: {name} has {len(params)} parameters, vk.xml {len(xparams)}')
        else:
            last_u32 = None
            for rust, p in zip(params, xparams):
                if why := check(rust, p, last_u32):
                    errors.append(f'{path}: {name} {p["name"]}: {rust}: {why}')
                if rust == 'u32' or (p['u32'] and rust == p['type']):
                    last_u32 = p['name']
    print('\n'.join(errors))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
