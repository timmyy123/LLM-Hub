package com.llmhub.llmhub.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import com.llmhub.llmhub.R

@Composable
fun KidModeSetPinDialog(
    onConfirm: (String) -> Unit,
    onDismiss: () -> Unit
) {
    var pin by remember { mutableStateOf("") }
    var error by remember { mutableStateOf("") }
    val pinErrorMessage = stringResource(R.string.kid_mode_pin_error)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.kid_mode_set_pin_title)) },
        text = {
            Column {
                Text(stringResource(R.string.kid_mode_set_pin_body))
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = pin,
                    onValueChange = { 
                        if (it.length <= 4 && it.all { char -> char.isDigit() }) {
                            pin = it
                            error = ""
                        }
                    },
                    label = { Text(stringResource(R.string.kid_mode_pin_label)) },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = error.isNotEmpty(),
                    supportingText = { if (error.isNotEmpty()) Text(error) }
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (pin.length == 4) {
                        onConfirm(pin)
                    } else {
                        error = pinErrorMessage
                    }
                }
            ) {
                Text(stringResource(R.string.kid_mode_enable))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.action_cancel))
            }
        }
    )
}

@Composable
fun KidModeVerifyPinDialog(
    onConfirm: (String) -> Unit,
    onDismiss: () -> Unit
) {
    var pin by remember { mutableStateOf("") }
    var error by remember { mutableStateOf("") }
    val pinErrorMessage = stringResource(R.string.kid_mode_pin_error)

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(stringResource(R.string.kid_mode_disable_title)) },
        text = {
            Column {
                Text(stringResource(R.string.kid_mode_verify_body))
                Spacer(modifier = Modifier.height(16.dp))
                OutlinedTextField(
                    value = pin,
                    onValueChange = { 
                        if (it.length <= 4 && it.all { char -> char.isDigit() }) {
                            pin = it
                            error = ""
                        }
                    },
                    label = { Text(stringResource(R.string.kid_mode_pin_label)) },
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    isError = error.isNotEmpty(),
                    supportingText = { if (error.isNotEmpty()) Text(error) }
                )
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    if (pin.length == 4) {
                        onConfirm(pin)
                    } else {
                        error = pinErrorMessage
                    }
                }
            ) {
                Text(stringResource(R.string.kid_mode_disable))
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text(stringResource(R.string.action_cancel))
            }
        }
    )
}
