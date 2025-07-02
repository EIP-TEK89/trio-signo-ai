[back](../../index.md)
# DELETE - Get Alphabet END
**URL:** `{root_url}/get-alphabet-end`

## Description

Cleanup image history after using [get-alphabet](get-alphabet.md) endpoint.

## Request

### Header
You must set in your request headers the following values:

```json
{
    "Content-Type": "multipart/form-data"
}
```

### Body
*None*

## Response
### 200
```json
{
    "message": "Sample history deleted"
}
```
### 400
> When the body or its values are incorrect.
```json
{
    "message": "Invalid IP address"
}
```
### 401
> Response defined in [index.md](../../index.md)

## Example
```sh
curl -X DELETE "{root_url}/get-alphabet" \
  -H "Content-Type: multipart/form-data"
```
