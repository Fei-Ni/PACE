"""CLI override parsing shared by the training entry point and the launchers (no torch dependency)."""


def normalize_dotlist_args(args):
    """
    Convert ['--x.y', 'val'] and ['--flag'] → ['x.y=val', 'flag=true']

    Multiple consecutive non-flag values after a key are assembled into an
    OmegaConf list override:
        ['--k', 'a', 'b', 'c'] → ['k=[a,b,c]']
    This handles bash array expansion of list-valued config fields.
    """
    normalized = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            if "=" in key:
                normalized.append(key)
                i += 1
            else:
                # Collect all following non-flag values
                j = i + 1
                values = []
                while j < len(args) and not args[j].startswith("--"):
                    values.append(args[j])
                    j += 1
                if not values:
                    normalized.append(f"{key}=true")
                elif len(values) == 1:
                    normalized.append(f"{key}={values[0]}")
                else:
                    list_str = ",".join(values)
                    normalized.append(f"{key}=[{list_str}]")
                i = j
        else:
            i += 1  # skip orphaned value
    return normalized
