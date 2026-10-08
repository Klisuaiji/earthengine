"""earthengine 命令行 (39)。

    earthengine generate --seed 12345 [--width --height --out --detail --stages]
    earthengine validate world/
    earthengine render world/ [--style]
    earthengine export world/ [--format]
    earthengine web --port 8899

现状旧版命令的 world/plates/ancient_map/info/export 语义映射：
    world → generate；plates → generate --stages tectonics；
    ancient_map → render --style ancient；info/export → export。
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def _cfg_from_args(args) -> dict:
    cfg = {"seed": args.seed, "width": args.width, "height": args.height,
           "detail": args.detail, "plate_model": args.plate_model}
    if args.config:
        import json
        if os.path.exists(args.config):
            with open(args.config) as f:
                cfg.update(json.load(f))
    if args.stages:
        cfg["stages"] = args.stages
    if args.out:
        cfg["out"] = args.out
    return cfg


def cmd_generate(args):
    cfg = _cfg_from_args(args)
    from earthengine.pipeline.stages import WorldGenerator
    from earthengine.pipeline.validator import WorldValidator
    from earthengine.io.export import export_world
    out = cfg.get("out", "world_out")
    t0 = time.time()
    gen = WorldGenerator(cfg)
    ws = gen.generate(stages=cfg.get("stages"))
    # 验证
    validator = WorldValidator(cfg)
    ok, report = validator.validate(ws)
    ws.validation = report
    print(f"[generate] seed={cfg['seed']} {ws.w}x{ws.h} "
          f"land={float(ws.ocean['land_mask'].mean()):.3f} "
          f"civ_pts={len(ws.civilization['points'])} "
          f"overall={report['scores']['overall']:.2f} OK={ok} "
          f"({time.time()-t0:.1f}s)")
    export_world(ws, out, render=True)
    print(f"[generate] exported -> {out}")
    return out


def cmd_validate(args):
    import numpy as np
    world_dir = args.world_dir
    meta_path = os.path.join(world_dir, "world.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: no world package at {world_dir}")
        return 1
    # 从 fields.npz 重建 WorldState 的快照以复用验证器
    from earthengine.pipeline.validator import WorldValidator
    from earthengine.io.npz import fields_dict
    # 用最小 WorldState 外壳承载场
    from earthengine.pipeline.world import WorldState
    import numpy as np
    z = np.load(os.path.join(world_dir, "fields.npz"))
    ws = WorldState(seed=1)
    h, w = z["terrain.elevation_macro"].shape
    ws.ocean = {"land_mask": z["ocean.land_mask"]}
    ws.boundaries = {"boundary_type": z.get("boundaries.type", None) if "boundaries.type" in z else None}
    ws.terrain = {"region": z["terrain.region"],
                  "elevation": z["terrain.elevation"] if "terrain.elevation" in z else z["terrain.elevation_macro"]}
    ws.plates = type("P", (), {"plates": z["plates.id"]})()
    ws.climate = {"temperature": z["atmosphere.temperature"]}
    ws.hydrology = {"uphill_segments": 0}
    v = WorldValidator()
    ok, report = v.validate(ws)
    print(f"[validate] overall={report['scores']['overall']:.2f} OK={ok}")
    for m in report["messages"]:
        print("  " + m)
    return 0 if ok else 1


def cmd_render(args):
    from earthengine.io.npz import fields_dict
    import numpy as np
    world_dir = args.world_dir
    if not os.path.exists(os.path.join(world_dir, "fields.npz")):
        print(f"ERROR: no world package at {world_dir}")
        return 1
    z = np.load(os.path.join(world_dir, "fields.npz"))
    from earthengine import rendering as R
    # 渲染风格
    style = args.style
    if style in ("ancient", "relief", "satellite"):
        elev = z.get("terrain.elevation")
        if elev is None:
            elev = z["terrain.elevation_macro"]
        biome = None
        if "vegetation.biome" in z:
            from earthengine.biology.vegetation import BIOME_OF_KOPPEN
            uniq = list(BIOME_OF_KOPPEN.values())
            idx = z["vegetation.biome"]
            names = np.array([list(BIOME_OF_KOPPEN.values())[i % len(BIOME_OF_KOPPEN)]
                              for i in range(int(idx.max())+1)])
            biome = names[idx]
        ice = z.get("climate.ice", None)
        out = os.path.join(world_dir, "render_%s.png" % style)
        if style == "satellite" and biome is not None:
            R.map.render_satellite(elev, biome, ice, out)
        elif style in ("ancient", "relief"):
            R.map.render_relief_hires(elev, out, land_mask=(elev >= 0), ice=ice)
        else:
            R.map.render_terrain_types(z["terrain.region"], out)
        print(f"[render] -> {out}")
    else:
        print(f"ERROR: unknown style {style}")
        return 1
    return 0


def cmd_export(args):
    import numpy as np
    world_dir = args.world_dir
    if not os.path.exists(os.path.join(world_dir, "fields.npz")):
        print(f"ERROR: no world package at {world_dir}")
        return 1
    fmt = args.format
    z = np.load(os.path.join(world_dir, "fields.npz"))
    from earthengine import rendering as R
    if fmt in ("png", "all"):
        out = os.path.join(world_dir, "export_%s.png" % fmt)
        R.map.render_float_field(z.get("terrain.elevation",
                                       z["terrain.elevation_macro"]), out, -8000, 4000)
        print(f"[export] -> {out}")
    elif fmt == "npy":
        out = os.path.join(world_dir, "export_elevation.npy")
        import numpy as np
        np.save(out, z["terrain.elevation_macro"])
        print(f"[export] -> {out}")
    else:
        print(f"ERROR: unknown format {fmt}")
        return 1
    return 0


def cmd_web(args):
    from earthengine.web import web_server
    web_server.run_server(port=args.port, host=args.host)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="earthengine",
                                description="EarthEngine v2.0 架空行星生成引擎")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="生成一颗行星")
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--width", type=int, default=1024)
    g.add_argument("--height", type=int, default=512)
    g.add_argument("--config", default=None)
    g.add_argument("--stages", default=None)
    g.add_argument("--out", default=None)
    g.add_argument("--detail", default="procedural", choices=["procedural", "diffusion"])
    g.add_argument("--plate-model", default="motion", choices=["motion", "legacy"])
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="验证 world 包")
    v.add_argument("world_dir")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("render", help="渲染 world 包")
    r.add_argument("world_dir")
    r.add_argument("--style", default="ancient", choices=["ancient", "relief", "satellite"])
    r.set_defaults(func=cmd_render)

    e = sub.add_parser("export", help="导出 world 包")
    e.add_argument("world_dir")
    e.add_argument("--format", default="png", choices=["png", "npy", "all"])
    e.set_defaults(func=cmd_export)

    w = sub.add_parser("web", help="启动一键 Web")
    w.add_argument("--port", type=int, default=8899)
    w.add_argument("--host", default="127.0.0.1")
    w.set_defaults(func=cmd_web)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
