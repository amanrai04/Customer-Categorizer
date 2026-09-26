#!/usr/bin/env bash
#
# Create the Azure resources the deployment workflow expects: a resource group,
# a container registry and a container-based web app.
#
# This is the scripted version of docs/automated_setup.md. It is idempotent:
# re-running it updates the existing resources instead of failing.
#
# Required environment variables:
#   AZURE_SUBSCRIPTION_ID   subscription to create the resources in
#   APP_NAME                globally unique prefix for every resource name
#
# Optional:
#   AZURE_LOCATION          region to create the resources in (default: eastus)
#   IMAGE_NAME              container image to deploy (default: customer-segmentation)

set -euo pipefail

: "${AZURE_SUBSCRIPTION_ID:?AZURE_SUBSCRIPTION_ID is required}"
: "${APP_NAME:?APP_NAME is required}"

AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
IMAGE_NAME="${IMAGE_NAME:-customer-segmentation}"

RESOURCE_GROUP="${APP_NAME}rg"
REGISTRY="${APP_NAME}acr"
WEB_APP="${APP_NAME}-web"
LOGIN_SERVER="${REGISTRY}.azurecr.io"
IMAGE="${LOGIN_SERVER}/${IMAGE_NAME}:latest"

if ! az account show >/dev/null 2>&1; then
    echo "Not signed in to Azure. Run 'az login' first." >&2
    exit 1
fi

echo "==> Resource group: ${RESOURCE_GROUP} (${AZURE_LOCATION})"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$AZURE_LOCATION" \
    --output none

echo "==> Container registry: ${REGISTRY}"
az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$REGISTRY" \
    --sku Basic \
    --location "$AZURE_LOCATION" \
    --admin-enabled true \
    --output none

REGISTRY_USERNAME="$(az acr credential show --name "$REGISTRY" --query username --output tsv)"
REGISTRY_PASSWORD="$(az acr credential show --name "$REGISTRY" --query passwords[0].value --output tsv)"

echo "==> Pushing ${IMAGE}"
docker login "$LOGIN_SERVER" --username "$REGISTRY_USERNAME" --password "$REGISTRY_PASSWORD"
docker build -t "$IMAGE" .
docker push "$IMAGE"

echo "==> Web app: ${WEB_APP}"
az webapp create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WEB_APP" \
    --container-image-name "$IMAGE" \
    --container-registry-server "$LOGIN_SERVER" \
    --container-registry-username "$REGISTRY_USERNAME" \
    --container-registry-password "$REGISTRY_PASSWORD" \
    --runtime-version "1.0" \
    --https-only true \
    --output none

echo "==> Platform settings"
az webapp config appsettings set \
    --resource-group "$RESOURCE_GROUP" \
    --name "$WEB_APP" \
    --settings \
        WEBSITES_PORT=5000 \
        WEBSITES_CONTAINER_START_TIME_LIMIT=60 \
        APP_PORT=5000 \
    --output none

cat <<EOF

Resources created.

Next:
  1. Add the application secrets to the web app:
       az webapp config appsettings set --resource-group ${RESOURCE_GROUP} \\
         --name ${WEB_APP} --settings MONGO_DB_URL=... AWS_ACCESS_KEY_ID=... \\
         AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=... \\
         CORS_ORIGINS=https://${WEB_APP}.azurewebsites.net

  2. Add the GitHub repository secrets listed in docs/deployment.md.

  3. Check the app:
       curl https://${WEB_APP}.azurewebsites.net/health
EOF
