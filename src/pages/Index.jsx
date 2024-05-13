import React, { useState } from "react";
import { Container, VStack, Text, Input, Button, HStack, Box, Image, IconButton } from "@chakra-ui/react";
import { FaRocket } from "react-icons/fa";

const Index = () => {
  const [lotteryNumbers, setLotteryNumbers] = useState([]);
  const [inputValue, setInputValue] = useState("");

  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleAddNumber = () => {
    if (inputValue && !isNaN(inputValue)) {
      setLotteryNumbers([...lotteryNumbers, inputValue]);
      setInputValue("");
    }
  };

  const handleClearNumbers = () => {
    setLotteryNumbers([]);
  };

  return (
    <Container centerContent maxW="container.md" height="100vh" display="flex" flexDirection="column" justifyContent="center" alignItems="center">
      <VStack spacing={4}>
        <Text fontSize="2xl">Quantum Lottery Predictor</Text>
        <Text>Enter your lottery numbers below:</Text>
        <HStack>
          <Input value={inputValue} onChange={handleInputChange} placeholder="Enter number" />
          <Button onClick={handleAddNumber}>Add</Button>
        </HStack>
        <Box>
          <Text>Numbers:</Text>
          <HStack spacing={2}>
            {lotteryNumbers.map((num, index) => (
              <Box key={index} p={2} borderWidth="1px" borderRadius="md">
                {num}
              </Box>
            ))}
          </HStack>
        </Box>
        <Button onClick={handleClearNumbers} colorScheme="red">
          Clear Numbers
        </Button>
        <IconButton aria-label="Launch Prediction" icon={<FaRocket />} size="lg" />
        <Image src="https://images.unsplash.com/photo-1617839625591-e5a789593135?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w1MDcxMzJ8MHwxfHNlYXJjaHwxfHxxdWFudHVtJTIwY29tcHV0ZXJ8ZW58MHx8fHwxNzE1NjM4NjAxfDA&ixlib=rb-4.0.3&q=80&w=1080" alt="Quantum Computer" />
      </VStack>
    </Container>
  );
};

export default Index;
