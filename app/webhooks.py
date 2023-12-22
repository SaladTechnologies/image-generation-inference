import config
import logging
import aiohttp
from monitoring import get_detailed_system_performance
import time


async def post_webhook(url, data):
    """Generic function to post data to a webhook URL asynchronously."""
    if url:
        try:
            data["node_info"] = {
                "identity": config.identity,
                "system_stats": get_detailed_system_performance(),  # Ensure this is async or doesn't block
            }
            # Include the UTC time as a datetime string
            data["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            headers = {}
            if config.webhook_auth["header"] and config.webhook_auth["value"]:
                headers[config.webhook_auth["header"]] = config.webhook_auth["value"]
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    response.raise_for_status()
                    logging.info(f"Webhook triggered successfully at {url}.")
        except aiohttp.ClientError as e:
            logging.error(f"Failed to trigger webhook at {url}. Error: {e}")


async def model_loaded(data):
    """Function to handle 'model loaded' event."""
    logging.debug("Handling model loaded event...")
    data["event"] = "model.loaded"
    await post_webhook(config.webhooks["model.loaded"], data)


async def model_unloaded(data):
    """Function to handle 'model unloaded' event."""
    logging.debug("Handling model unloaded event...")
    data["event"] = "model.unloaded"
    await post_webhook(config.webhooks["model.unloaded"], data)


async def image_stored(data):
    """Function to handle 'image stored' event."""
    logging.debug("Handling image stored event...")
    data["event"] = "image.stored"
    await post_webhook(config.webhooks["image.stored"], data)
