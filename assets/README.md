# Bobby

Bobby is bobcat's mascot, a bobcat drawn as pixel art. `tools/mascot.py`
draws both files here from text pixel grids:

```console
$ cd tools && uv run python mascot.py ../assets
```

The script needs `rsvg-convert` from Homebrew's `librsvg` to render the
social preview card.

| File | Use |
|---|---|
| `banner.svg` | The README banner, where Bobby blinks, smiles, thinks, and dozes |
| `social-preview.png` | The 1280 by 640 card GitHub shows when someone shares the repository |

A light rim keeps the outlines visible on dark backgrounds. GitHub takes
the social preview card from the repository's settings page, under
Social preview.

## Colors

The mascot's colors double as the project's colors. Benchmark charts
draw bobcat in the fur color and every other engine in a gray.

| Name | Hex | Use |
|---|---|---|
| Fur | `#c28d5a` | bobcat's bars in charts |
| Outline | `#2b1a10` | Text and chart axes |
| Markings | `#7a4724` | Secondary text and highlights |
| Cream | `#f4e4c6` | Backgrounds |
| Amber | `#e0b53c` | Accents |
| Gray 1 to 4 | `#5f5f5f`, `#7f7f7f`, `#9f9f9f`, `#bfbfbf` | Other engines in charts |

## License

Bobby is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Anyone may use, change, and share Bobby, including for stickers, slides,
and fan art, with credit to the bobcat project.
