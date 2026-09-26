"""Names of the environment variables read by the application."""

# MongoDB Atlas connection string, e.g.
# ``mongodb+srv://<user>:<password>@<cluster>.mongodb.net``
MONGODB_URL_KEY: str = "MONGO_DB_URL"

# AWS credentials used to talk to S3.
AWS_ACCESS_KEY_ID_ENV_KEY: str = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY: str = "AWS_SECRET_ACCESS_KEY"
AWS_DEFAULT_REGION_ENV_KEY: str = "AWS_DEFAULT_REGION"

#: Region used when ``AWS_DEFAULT_REGION`` is not set.
DEFAULT_REGION: str = "ap-south-1"

#: Comma separated list of origins allowed by the CORS middleware. ``*`` means
#: "any origin" and should only be used for local development.
CORS_ORIGINS_ENV_KEY: str = "CORS_ORIGINS"
