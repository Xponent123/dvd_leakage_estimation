"""Static CMOS topology and complete-circuit DC reference simulation.

Eight total received bits; XOR=1 detects odd parity in an even-parity code.
All sizes are explicit engineering choices authorized by the team.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess

from .spice import OPTIONS, DEFAULT_MODEL, ROOT, sha256

WN = 64e-9
WP = 128e-9
LENGTH = 32e-9
GATES = ('INV', 'NAND2', 'NOR2', 'AND2', 'OR2')
PROTOCOL = 'static-cmos-power-v1'


@dataclass(frozen=True)
class FET:
    name: str
    kind: str
    d: str
    g: str
    s: str
    b: str

    @property
    def w(self):
        return WN if self.kind == 'nmos' else WP

    def line(self):
        return f'M{self.name} {self.d} {self.g} {self.s} {self.b} {self.kind} L={LENGTH:.17g} W={self.w:.17g}'


@dataclass(frozen=True)
class Cell:
    name: str
    gate: str
    inputs: tuple[str, ...]
    output: str


def cell_fets(cell: Cell) -> list[FET]:
    a = cell.inputs[0]
    b = cell.inputs[-1]
    out = cell.output
    x, z = cell.name+'_x', cell.name+'_z'
    def m(suffix, kind, d, g, s):
        return FET(cell.name+'_'+suffix, kind, d, g, s, '0' if kind == 'nmos' else 'vdd')
    if cell.gate == 'INV':
        return [m('n','nmos',out,a,'0'), m('p','pmos',out,a,'vdd')]
    inv = cell.gate in ('AND2', 'OR2')
    mid = z if inv else out
    if cell.gate in ('NAND2','AND2'):
        devices = [m('na','nmos',mid,a,x),m('nb','nmos',x,b,'0'),
                   m('pa','pmos',mid,a,'vdd'),m('pb','pmos',mid,b,'vdd')]
    elif cell.gate in ('NOR2','OR2'):
        devices = [m('na','nmos',mid,a,'0'),m('nb','nmos',mid,b,'0'),
                   m('pa','pmos',x,a,'vdd'),m('pb','pmos',mid,b,x)]
    else:
        raise ValueError(cell.gate)
    if inv:
        devices += [m('ni','nmos',out,z,'0'),m('pi','pmos',out,z,'vdd')]
    return devices


def gate_cell(gate: str) -> Cell:
    return Cell('g', gate, ('a',) if gate == 'INV' else ('a','b'), 'out')


def parity_cells() -> list[Cell]:
    cells = []
    def xor(name,a,b,out):
        t,u,v = name+'_t',name+'_u',name+'_v'
        cells.extend([Cell(name+'_0','NAND2',(a,b),t),
                      Cell(name+'_1','NAND2',(a,t),u),
                      Cell(name+'_2','NAND2',(b,t),v),
                      Cell(name+'_3','NAND2',(u,v),out)])
    for i in range(4):
        xor(f'x{i}',f'a{2*i}',f'a{2*i+1}',f'p{i}')
    xor('x4','p0','p1','p4')
    xor('x5','p2','p3','p5')
    xor('x6','p4','p5','out')
    return cells


def logic(gate: str, bits: tuple[int,...]) -> int:
    a,b = bits[0],bits[-1]
    return {'INV':1-a,'NAND2':1-(a&b),'NOR2':1-(a|b),'AND2':a&b,'OR2':a|b}[gate]


def propagate(vector: int):
    nodes = {f'a{i}':(vector>>i)&1 for i in range(8)}
    states = []
    for cell in parity_cells():
        bits = tuple(nodes[n] for n in cell.inputs)
        nodes[cell.output] = logic(cell.gate,bits)
        states.append((cell,bits))
    if nodes['out'] != vector.bit_count()%2:
        raise AssertionError('Boolean parity topology mismatch')
    return nodes, states


def process_model(corner: dict, model: Path = DEFAULT_MODEL) -> str:
    text = model.read_text()
    blocks = re.split(r'(?im)(?=^\s*\.model\s)', text)
    for index,block in enumerate(blocks):
        match = re.match(r'(?i)\s*\.model\s+(nmos|pmos)\s+\1\b',block)
        if not match:
            continue
        prefix = match.group(1)[0]+'_'
        for name in ('vth0','toxe','ndep','u0','xj'):
            block,count = re.subn(rf'(?i)(\b{name}\s*=\s*)[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?',
                lambda m:m.group(1)+format(float(corner[prefix+name]),'.17g'),block)
            if count != 1:
                raise ValueError(f'Expected one {prefix}{name}, found {count}')
        blocks[index] = block
    return ''.join(blocks)


def circuit_deck(corner: dict, kind: str, vectors: list[int] | None = None):
    if kind == 'PARITY8':
        cells = parity_cells()
        inputs = tuple(f'a{i}' for i in range(8))
    else:
        cells = [gate_cell(kind)]
        inputs = cells[0].inputs
    devices = [fet for cell in cells for fet in cell_fets(cell)]
    if vectors is None:
        vectors = list(range(1<<len(inputs)))
    # Gate integer bit order is a=MSB,b=LSB. Parity integers use a0=LSB.
    def bits(v):
        return tuple((v>>i)&1 for i in (range(len(inputs)) if kind=='PARITY8' else reversed(range(len(inputs)))))
    nodes = sorted({n for d in devices for n in (d.d,d.g,d.s,d.b)}-{'0','vdd',*inputs})
    observed = ['out'] if kind == 'PARITY8' else nodes
    vdd = float(corner['vdd'])
    lines = [f'LUSR {kind}; corner={corner.get("corner_id","nominal")}; external-source power/VDD',
             '* Derived model: only assigned five process parameters per polarity replaced.',
             process_model(corner), OPTIONS, f'.temp {float(corner["temp"]):.17g}',
             f'Vdd vdd 0 {vdd:.17g}',
             *[f'Vin{i} {n} 0 0' for i,n in enumerate(inputs)],
             *[d.line() for d in devices], '.control','set numdgt=17','set noaskquit']
    for vector in vectors:
        b = bits(vector)
        lines += [f'alter Vin{i} dc = {bit*vdd:.17g}' for i,bit in enumerate(b)]
        source_terms = ['i(Vdd)']+[f'i(Vin{i})' for i,bit in enumerate(b) if bit]
        lines += ['op',f'echo LUSR_CIRCUIT {vector}',
                  'let leakage = -('+'+'.join(source_terms)+')',
                  'let supply = -i(Vdd)', 'print leakage supply',
                  *[f'print v({n})' for n in observed], 'destroy all']
    return '\n'.join(lines+['quit','.endc','.end','']),observed


def simulate_corner(corner:dict,kind:str,exe:Path,directory:Path,vectors=None,cache=True):
    directory.mkdir(parents=True,exist_ok=True)
    deck,nodes = circuit_deck(corner,kind,vectors)
    count = 256 if kind=='PARITY8' else (2 if kind=='INV' else 4)
    vectors = list(range(count)) if vectors is None else list(vectors)
    identity = hashlib.sha256((PROTOCOL+sha256(exe)+deck).encode()).hexdigest()
    target = directory/f'{kind.lower()}_{int(corner["corner_id"]):03d}'
    cachefile = target.with_suffix('.json')
    if cache and cachefile.exists():
        saved=json.loads(cachefile.read_text())
        if saved.get('identity')==identity and len(saved.get('rows',[]))==len(vectors):
            return saved['rows']
    target.with_suffix('.cir').write_text(deck)
    result = subprocess.run([str(exe),'-n','-b',str(target.with_suffix('.cir').resolve())],
                            cwd=directory,capture_output=True,text=True,timeout=180)
    log=result.stdout+'\nSTDERR:\n'+result.stderr
    target.with_suffix('.log').write_text(log)
    if result.returncode or re.search(r'(?im)^\s*(?:error\b|fatal\b)|simulation\(s\) aborted|operating point failed',log):
        raise RuntimeError(f'Circuit simulation failed: {target}.log')
    rows={}
    current=None
    for line in result.stdout.splitlines():
        match=re.fullmatch(r'\s*LUSR_CIRCUIT\s+(\d+)\s*',line)
        if match:
            current=int(match.group(1))
            if current in rows:
                raise RuntimeError('Duplicate vector')
            rows[current]={}
        match=re.fullmatch(r'\s*(leakage|supply|v\([^)]+\))\s*=\s*(\S+)\s*',line)
        if match and current is not None:
            rows[current][match.group(1)]=float(match.group(2))
    if set(rows)!=set(vectors):
        raise RuntimeError(f'Incomplete simulation: {target}')
    records=[]
    for vector in vectors:
        values=rows[vector]
        if set(values)!={'leakage','supply',*[f'v({n})' for n in nodes]}:
            raise RuntimeError(f'Missing measurements: {target} vector {vector}')
        import math
        if not all(math.isfinite(v) for v in values.values()) or values['leakage']<=0:
            raise RuntimeError(f'Nonfinite/nonpositive power: {target} vector {vector}')
        bits=tuple((vector>>i)&1 for i in reversed(range(1 if kind=='INV' else 2)))
        expected=vector.bit_count()%2 if kind=='PARITY8' else logic(kind,bits)
        out=values['v(out)']/float(corner['vdd'])
        if (out>=.5)!=bool(expected) or (expected==0 and out>.2) or (expected==1 and out<.8):
            raise RuntimeError(f'Logic level failed: {target} vector {vector} output/VDD={out}')
        record={'corner_id':int(corner['corner_id']),'circuit':kind,'vector':vector,
                'input_bits':format(vector,f'0{8 if kind=="PARITY8" else len(bits)}b'),
                'vdd':float(corner['vdd']),'temp':float(corner['temp']),
                'spice_leakage_A':values['leakage'],'spice_supply_A':values['supply'],
                'spice_power_W':values['leakage']*float(corner['vdd']),
                'output_V':values['v(out)'],'expected_output':expected,
                'node_voltages_json':json.dumps({n:values[f'v({n})'] for n in nodes},sort_keys=True)}
        records.append(record)
    cachefile.write_text(json.dumps({'identity':identity,'rows':records},indent=2)+'\n')
    return records
