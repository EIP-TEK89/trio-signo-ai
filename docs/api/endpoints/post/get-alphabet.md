[back](../../index.md)
# POST - Get Alphabet
**URL:** `{root_url}/get-alphabet`

## Description

Gets an image and return the letter it represent in LSF (French Sign Language) or nothing if it's not a letter.
**IMPORTANT** this endpoint remembers previous image sent in case you want to send a video streaming. Therefore **you must** call [get-alphabet-end](get-alphabet-end.md) endpoint when finishing using this endpoint.

## Request

### Header
You must set in your request headers the following values:

```json
{
    "Content-Type": "multipart/form-data"
}
```

### Body
The request body should be in the **form-data** format and include the following parameters:

| Key  | Type   | Description                                              |
|------|--------|----------------------------------------------------------|
| file | File   | The image file to be processed. (in .jpg or .png format) |

## Response
### 200
```json
{
    "message": "A"
}
```
Or in case no hand is found:
```json
{
    "message": null
}
```
### 400
> When the body or its values are incorrect.
```json
{
    "message": "Error message"
}
```
### 401
> Response defined in [index.md](../../index.md)

## Example
```sh
curl -X POST "{root_url}/get-alphabet" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@B.jpg"
```
