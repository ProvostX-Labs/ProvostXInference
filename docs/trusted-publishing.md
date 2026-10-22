# NuGet Trusted Publishing — Setup Checklist (one-time, owner)

Trusted publishing means the GitHub Actions workflow gets a **short-lived key via OIDC**
at push time — no API key is ever stored in GitHub secrets, nothing to leak or rotate.
nuget.org verifies the workflow's identity against a Trusted Publisher policy you
register once on the package.

## 1. What YOU do on nuget.org (one-time, ~2 minutes)

1. Sign in at nuget.org → your account (the one that owns the existing
   `ProvostXInference` package ID).
2. Go to **Account settings → Trusted Publishers** (or, from the package's
   management page: **Manage → Trusted Publishers**).
3. Click **Add trusted publisher** and fill EXACTLY:
   - **Package owner:** your nuget.org account/owner under which the package ID
     `ProvostXInference` is registered
   - **Package ID:** `ProvostXInference` (exact, case-sensitive)
   - **Repository type:** `GitHub`
   - **Repository owner:** `ProvostX-Labs`
   - **Repository name:** `ProvostXInference`
   - **Workflow file:** `ci.yml` (filename ONLY, no path, no `.github/workflows/` —
     this is the #1 typo; it must match the committed filename exactly)
   - **Environment:** leave EMPTY (our workflow does not use a GitHub environment —
     if you set one here, every push fails policy validation)
4. Save. Done — there is no key, no secret, no rotation.

### If the package ID is owned by an organization
The trusted publisher binds to the OWNER of the package. If `ProvostXInference` on
nuget.org is under an organization account, register the policy while signed in with
that owner account (or as an owner of it). The policy owner must match the pusher.

## 2. What YOU do on GitHub (one-time, ~1 minute)

One repository secret is required by the login action (it identifies WHO logs in —
it is not an API key):

- **Repo → Settings → Secrets and variables → Actions → New repository secret**
  - Name: `NUGET_USER`
  - Value: your **nuget.org username** (the account name, not email)

## 3. What the workflow does (already wired, committed locally)

- `ci.yml` → `nuget-publish` job (needs `nuget-pack`), `permissions: id-token: write`
- `NuGet/login@v1` exchanges the workflow's OIDC token for a ~1-hour ephemeral key
- `dotnet nuget push` publishes with that ephemeral key; `--skip-duplicate` guards reruns
- Version comes from the `package_version` dispatch input (default 0.1.0)

## 4. How a release actually runs (when you order it)

1. Push the repo to `github.com/ProvostX-Labs/ProvostXInference` (explicit order required).
2. Actions tab → **CI** → **Run workflow** → set `package_version` → Run.
3. All lanes build + test; `nuget-pack` assembles every RID slice into the nupkg;
   `nuget-publish` logs in via OIDC and pushes.
4. If the push fails with `409`/`policy` errors: 99% it is a field mismatch vs the
   Trusted Publisher entry (workflow filename, owner, or an environment set on one
   side only). Compare the policy fields against the job's context — nothing else
   in the chain uses identity.

## 5. Security notes

- No long-lived secret exists anywhere in the chain — a leaked runner token expires
  within the hour by construction.
- The publish job runs ONLY on explicit `workflow_dispatch` — never on push/PR —
  so a PR cannot trigger a publish.
- First-time policy note: nuget.org requires the package ID to exist (yours does).
  The policy applies from the NEXT push; the very first trusted push may need the
  package's existing owner to confirm — follow the nuget.org UI if it prompts.
