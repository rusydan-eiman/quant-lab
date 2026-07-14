# Deployment Architecture (Long-term Plan)

## Current State (2026-07)

- **No deployment** — local development only
- **Cost**: $0/month
- **Reason**: User is still learning, no paying users, no income

## Phase 1: First Money → Simple VPS

**When**: After getting first job or stable income ($50+/month disposable)

**Stack**:
- VPS: Hetzner / Vultr / DigitalOcean — $5-10/month
- Docker Compose: 1 file, 4 services
  - `postgres` (database)
  - `app` (quant-lab API + worker)
  - `nginx` (reverse proxy)
  - `backup` (daily DB dump)
- Cron: daily at 6am, runs `python -m src.main`
- Storage: 20GB volume for DB + logs

**Why this first**:
- $10/month is affordable
- One YAML file (vs K8s's hundreds of lines)
- Easy to debug
- Easy to learn

## Phase 2: Quant-Lab Becomes Reliable → Managed K8s

**When**: User actively uses quant-lab for personal stock trading + strategy is validated

**Stack**:
- Managed K8s: DigitalOcean Kubernetes / GKE / EKS — $30-50/month for small cluster
- Managed Postgres: DigitalOcean Managed DB / Cloud SQL — $15-25/month
- (Optional) Redis for caching — $10/month
- S3-compatible object storage for backups — $5/month

**Why upgrade from Compose**:
- Multiple services need to scale independently
- Zero-downtime deploys (rolling updates)
- Better resource isolation
- Auto-scaling for burst workloads

**Why managed (not self-hosted K8s)**:
- Self-hosted K8s is operationally expensive
- Control plane upgrades alone are hours of work
- Managed = they do the hard stuff

## Phase 3: User Growth (Far Future)

**When**: Other people start using quant-lab

**Stack additions**:
- Multi-region K8s (GKE or EKS)
- Managed Redis cluster
- CDN for static assets
- Observability: Prometheus + Grafana + Datadog
- CI/CD: GitHub Actions → automated deploys
- Secret management: HashiCorp Vault or AWS Secrets Manager

## Why Not Supabase

For **quant-lab**:
- Supabase free tier: 500MB database, pauses after 1 week inactivity
- Supabase Pro: $25/month — same price as a small VPS that runs the whole stack
- Managed Postgres gives more control, easier to backup, better for time-series data
- Supabase is great for B2C apps with auth + storage; quant-lab just needs a DB

For **retail-renaissance** (different project):
- Supabase IS good here (auth + DB + storage in one)
- Could use Supabase when this deploys

## Migration Path

```
Phase 0: Local SQLite (free)
        ↓
Phase 1: VPS + Docker Compose + Postgres ($10/mo)
        ↓
Phase 2: Managed K8s + Managed Postgres ($50/mo)
        ↓
Phase 3: Multi-region K8s + observability ($200+/mo)
```

Each phase adds capability, costs more, requires more skills to operate. Don't skip phases.

## References

- Hetzner VPS pricing: https://www.hetzner.com/cloud
- DigitalOcean K8s pricing: https://www.digitalocean.com/products/kubernetes
- Docker Compose docs: https://docs.docker.com/compose/
- Migration Compose → K8s guide: https://kubernetes.io/docs/tasks/configure-pod-container/translate-compose-kubernetes/

## Decision Date

2026-07-14 — saved for future reference