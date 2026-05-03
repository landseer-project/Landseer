# post_magnet (MagNet) — artifact Dockerfile

**Upstream:** https://github.com/Trevillie/MagNet  
**Pinned commit:** `b115cf8798dc0c304ca0da34b2bdf1485da6f54f` (`upstream/master` at patch generation time)

**Landseer delta:** `patches/0001-*.patch` … `git format-patch` series, applied with `git am` in the image build (see `Dockerfile`).

Build:

```bash
docker build -f Dockerfile -t post_magnet:artifact .
```
