# Deployment

Deployment is optional. The `ci.yml` workflow runs on every push and pull
request and needs no configuration. `deploy.yml` only runs once the repository
secrets below are set, so a fork or a clone can use CI alone.

## 1. Deploy the application

```bash
python app.py
```

The app listens on <http://localhost:5000> by default. Override with `APP_HOST`
and `APP_PORT`.

## 2. Build and run locally with Docker

```bash
docker build -t customer-segmentation .

docker run --rm -p 5000:5000 --env-file .env customer-segmentation
```

A minimal `.env`:

```dotenv
MONGO_DB_URL=mongodb+srv://user:password@cluster.mongodb.net
AWS_ACCESS_KEY_ID=your-key-id
AWS_SECRET_ACCESS_KEY=your-secret
AWS_DEFAULT_REGION=ap-south-1
```

Never commit a populated `.env`; it is already in `.gitignore`.

## 3. Deploy to Azure

Create the Azure resources first - either
[`manual_setup.md`](manual_setup.md) for the portal walkthrough or
[`automated_setup.md`](automated_setup.md) for the CLI version. Both end with a
resource group, a container registry and a container-based web app.

Then add the repository secrets below under **Settings -> Secrets and
variables -> Actions**.

### Repository secrets

| Secret | Where to get it |
|--------|-----------------|
| `AZURE_CREDENTIALS` | The whole JSON blob from `az ad sp create-for-rbac --sdk-auth` |
| `AZURE_LOGIN_SERVER` | **Login server** on the container registry's *Access keys* page, e.g. `myappacr.azurecr.io` |
| `REGISTRY_USERNAME` | **Username** on the same page |
| `REGISTRY_PASSWORD` | Either **Password** on the same page |
| `REPO_NAME` | The image name to push, e.g. `customer-segmentation` |
| `AZURE_WEB_APP_NAME` | The web app name, e.g. `myapp-web` |

### Application settings

These are set on the web app, not in GitHub, so that the running container can
read them. See step 5 of either setup guide. At minimum:

| Setting | Value |
|---------|-------|
| `WEBSITES_PORT` | `5000` |
| `APP_PORT` | `5000` |
| `MONGO_DB_URL` | your MongoDB connection string |
| `AWS_ACCESS_KEY_ID` | your S3 access key |
| `AWS_SECRET_ACCESS_KEY` | your S3 secret |
| `AWS_DEFAULT_REGION` | e.g. `ap-south-1` |
| `CORS_ORIGINS` | the app's own URL |

## 4. What the deploy workflow does

`.github/workflows/deploy.yml` triggers on a push to `main` that touches
application code, and can also be started by hand from the **Actions** tab. It:

1. checks out the repository
2. signs in to Azure with `AZURE_CREDENTIALS`
3. signs in to the container registry
4. builds the image and pushes it tagged with the commit SHA and `latest`
5. points the web app at the new image

Changes limited to `docs/`, `notebooks/` or Markdown files do not trigger a
deploy.

> The workflow targets the `production` environment. Create it under **Settings
> -> Environments** if you want an approval gate before deploys.

## 5. Verify

```bash
curl https://<your-web-app-name>.azurewebsites.net/health
# {"status":"ok"}
```

Then open the app URL, submit the prediction form, and call `GET /train` once to
confirm the container can reach MongoDB and S3.

## 6. Tear down

```bash
az group delete --resource-group <your-app-name>rg --yes
```
