# Manual Azure setup

This guide creates the Azure resources the deployment workflow expects, using
the Azure portal. If you would rather script it, use
[`automated_setup.md`](automated_setup.md) instead - the two produce the same
resources.

The deployment target is an **Azure Web App running a Docker container** backed
by an image in an **Azure Container Registry**.

## What gets created

| Resource | Purpose |
|----------|---------|
| Resource group | Holds everything below, so it can be deleted in one step |
| Container registry | Stores the application image |
| Web app | Runs the container |
| Service principal | Lets GitHub Actions deploy without a human in the loop |

Naming used throughout this guide, with `{name}` as your own prefix:

| Setting | Value |
|---------|-------|
| Resource group | `{name}rg` |
| Registry | `{name}acr` |
| Web app | `{name}-web` |

> Azure requires resource names to be globally unique. If `{name}acr` is
> already taken, pick another prefix.

## 1. Prerequisites

You need an Azure subscription. If you do not have one, create a free account
at <https://portal.azure.com> - the free tier includes enough credit to build
and test this setup.

Sign in to the [Azure portal](https://portal.azure.com) and confirm the
subscription you want to bill is selected in the top toolbar.

## 2. Create the resource group

1. In the portal search box, type **Resource group** and open it.
2. Click **Create**.
3. Fill in:
   - **Subscription**: the one you want to bill
   - **Resource group name**: `{name}rg`
   - **Region**: pick the one closest to your users
4. Click **Review + create**, then **Create**.

## 3. Create the container registry

1. Open the resource group `{name}rg` and click **Create**.
2. Search for **Container Registry** and open it.
3. Click **Create** and fill in:
   - **Subscription** and **Resource group**: `{name}rg`
   - **Registry name**: `{name}acr`
   - **Location**: the same region as the resource group
   - **SKU**: `Basic`
4. Click **Review + create**, then **Create**.
5. After it is created, click **Go to resource** and open
   **Services -> Access keys**.
6. Set **Admin user** to **Enabled**, then copy:
   - **Login server** - looks like `{name}acr.azurecr.io`
   - **Username**
   - **Password**

> Admin access is fine for a demo. For anything real, use a managed identity or
> a service principal instead of a shared admin password.

## 4. Create the web app

The image has to exist in the registry before the web app can pull it, so build
and push it first from a machine that has Docker:

```bash
export AZURE_LOGIN_SERVER="{name}acr.azurecr.io"
export REGISTRY_USERNAME="<username from the access keys page>"
export REGISTRY_PASSWORD="<password from the access keys page>"

docker login "$AZURE_LOGIN_SERVER" -u "$REGISTRY_USERNAME" -p "$REGISTRY_PASSWORD"
docker build -t "$AZURE_LOGIN_SERVER/customer-segmentation:latest" .
docker push "$AZURE_LOGIN_SERVER/customer-segmentation:latest"
```

Now create the web app in the portal:

1. Open the resource group `{name}rg` and click **Create**.
2. Search for **Container Web App** and open it.
3. Click **Create** and fill in:
   - **Basics**
     - **Subscription** and **Resource group**: `{name}rg`
     - **Name**: `{name}-web`
   - **Container**
     - **Image type**: `Container Registry`
     - **Image**: `customer-segmentation:latest` from `{name}acr`
4. Click **Review + create**, then **Create**. The first deployment takes a few
   minutes while the image is pulled.

## 5. Add the application settings

The container listens on port 5000, and the container-based web app needs to be
told about it:

1. Open the web app, then **Configuration -> Environment variables**.
2. Click **Add** and create:

   | Name | Value |
   |------|-------|
   | `WEBSITES_PORT` | `5000` |
   | `WEBSITES_CONTAINER_START_TIME_LIMIT` | `60` |
   | `APP_PORT` | `5000` |
   | `MONGO_DB_URL` | your MongoDB connection string |
   | `AWS_ACCESS_KEY_ID` | your S3 access key |
   | `AWS_SECRET_ACCESS_KEY` | your S3 secret |
   | `AWS_DEFAULT_REGION` | e.g. `ap-south-1` |
   | `CORS_ORIGINS` | `https://{name}-web.azurewebsites.net` |

   > `WEBSITES_CONTAINER_START_TIME_LIMIT` is in seconds. Raise it if the image
   > is large and the container is killed during start-up.
   >
   > Set `CORS_ORIGINS` to the app's own URL. The default `*` is convenient
   > locally but should not be used in production.
3. Click **Save**, then **Continue**.

## 6. Verify the deployment

Open **Overview** and copy the URL under **URL**, then:

```bash
curl https://{name}-web.azurewebsites.net/health
# {"status":"ok"}
```

Submit the form at `https://{name}-web.azurewebsites.net/` to confirm a
prediction comes back, and run `GET /train` once to confirm the pipeline can
reach MongoDB and S3 from inside the container.

## 7. Clean up

Delete the whole resource group when you are done - this removes the registry,
the web app and everything else created above.

**Resource groups -> `{name}rg` -> Delete resource group** in the portal, or:

```bash
az group delete --resource-group {name}rg --yes
```

Or use the helper script:

```bash
bash scripts/delete_azure_resources.sh
```

Next: [`deployment.md`](deployment.md) to automate the deploy step with GitHub
Actions.
